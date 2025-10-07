import os
import subprocess
import time
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, Response

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './web_tmp_dir'
# Store running processes
running_processes = {}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
DATA_DIR = "/ime/hdd/rhaas/SUP-5301/database"

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def stream_output(process, pdb_file, protein_id, prediction_id):
    """Stream output from subprocess to client."""
    def generate():
        # Stream stdout line-by-line in text mode
        for line in iter(process.stdout.readline, ''):
            line_str = line.rstrip('\n')
            if line_str:
                yield f"data: {json.dumps({'type': 'output', 'message': line_str})}\n\n"
        
        # Check for errors
        process.wait()
        
        # Clean up the process from the running processes
        if prediction_id in running_processes:
            del running_processes[prediction_id]
            
        if process.returncode != 0:
            try:
                error = process.stderr.read()
                yield f"data: {json.dumps({'type': 'error', 'message': error})}\n\n"
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred while reading the error output'})}\n\n"
        else:
            # Send completion message with PDB file info
            yield f"""data: {json.dumps({
                'type': 'complete',
                'pdb_file': pdb_file,
                'protein_id': protein_id
            })}\n\n"""
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # for nginx
            'Connection': 'keep-alive'
        }
    )

@app.route('/cancel_prediction', methods=['POST'])
def cancel_prediction():
    prediction_id = request.json.get('prediction_id')
    if not prediction_id:
        return jsonify({'status': 'error', 'message': 'No prediction ID provided'}), 400
        
    if prediction_id in running_processes:
        process_info = running_processes[prediction_id]
        try:
            # Terminate the process group to ensure all child processes are killed
            if process_info['process'].poll() is None:  # Check if process is still running
                try:
                    # Try to terminate gracefully first
                    os.killpg(os.getpgid(process_info['process'].pid), 15)  # SIGTERM
                    
                    # Wait a bit for the process to terminate
                    try:
                        process_info['process'].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't terminate
                        os.killpg(os.getpgid(process_info['process'].pid), 9)  # SIGKILL
                        
                    return jsonify({
                        'status': 'cancelled',
                        'message': 'Prediction was successfully cancelled'
                    })
                except ProcessLookupError:
                    # Process already terminated
                    pass
                except Exception as e:
                    return jsonify({
                        'status': 'error',
                        'message': f'Error cancelling prediction: {str(e)}'
                    }), 500
            
            # Clean up
            del running_processes[prediction_id]
            return jsonify({'status': 'already_done', 'message': 'Process was already terminated'})
            
        except Exception as e:
            # Clean up even if there was an error
            if prediction_id in running_processes:
                del running_processes[prediction_id]
            return jsonify({
                'status': 'error',
                'message': f'Error cancelling prediction: {str(e)}'
            }), 500
    
    return jsonify({'status': 'not_found', 'message': 'No running prediction found with that ID'})

@app.route('/process', methods=['GET', 'POST'])
def process():
    prediction_id = None
    if request.method == 'GET':
        # Handle SSE connection
        sequence = request.args.get('sequence', '').strip()
        description = request.args.get('description', '').strip()
        residue_idx = int(request.args.get('residue_idx', 1))
        protein_id = request.args.get('protein_id', 'demo').strip()
        prediction_id = request.args.get('prediction_id', None)
    else:
        # Handle form submission
        sequence = request.form.get('sequence', '').strip()
        description = request.form.get('description', '').strip()
        protein_id = request.form.get('protein_id', 'demo').strip()
        prediction_id = request.form.get('prediction_id', None)
        try:
            residue_idx = int(request.form.get('residue_idx', 1))
        except (ValueError, TypeError):
            residue_idx = 1  # Default value if conversion fails
    
    if sequence == '':
        return jsonify({'error': 'Protein sequence is required'}), 400
    
    # Format FASTA content
    fasta_content = f">{protein_id} {description}\n{sequence}"
    
    # Run OpenFold in a subprocess with streaming output
    fasta_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'fasta_{protein_id}')
    os.makedirs(fasta_dir, exist_ok=True)
    
    # Save FASTA file
    fasta_path = os.path.join(fasta_dir, f"{protein_id}.fasta")
    with open(fasta_path, 'w') as f:
        f.write(fasta_content)
    
    # Define output directories
    attn_map_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/attention_files_{protein_id}_demo_tri_{residue_idx}')
    output_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}')
    image_output_dir = os.path.join(os.path.abspath(app.config['UPLOAD_FOLDER']), f'outputs/attention_images_{protein_id}_demo_tri_{residue_idx}')
    
    data_dir = os.path.realpath(os.path.expanduser(DATA_DIR))
    # Create output directories
    os.makedirs(attn_map_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    
    #Build the command
    cmd = [
        'python3', '-u', 'run_pretrained_openfold.py',
        fasta_dir,
        f'{data_dir}/pdb_mmcif/mmcif_files',
        '--output_dir', output_dir,
        '--config_preset', 'model_1_ptm',
        '--uniref90_database_path', f'{data_dir}/uniref90/uniref90.fasta',
        '--mgnify_database_path', f'{data_dir}/mgnify/mgy_clusters_2022_05.fa',
        '--pdb70_database_path', f'{data_dir}/pdb70/pdb70',
        '--uniclust30_database_path', f'{data_dir}/uniclust30/uniclust30_2018_08/uniclust30_2018_08',
        '--bfd_database_path', f'{data_dir}/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
        '--save_outputs',
        '--model_device', 'cuda:0',
        '--attn_map_dir', attn_map_dir,
        '--num_recycles_save', '1',
        '--triangle_residue_idx', str(residue_idx),
        '--demo_attn'
    ]

    # Store the process
    process = subprocess.Popen(
        cmd,
        cwd='..',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr into stdout to prevent blocking
        text=True,
        bufsize=1,  # line buffered
        universal_newlines=True,
        start_new_session=True  # Required for process group management
    )
    
    # Store the process in the global dictionary
    running_processes[prediction_id] = {
        'process': process,
        'start_time': time.time()
    }
    
    # Path to the expected PDB file
    pdb_file = f'outputs/my_outputs_align_{protein_id}_demo_tri_{residue_idx}/predictions/{protein_id}_model_1_ptm_relaxed.pdb'
    # Return the streaming response
    return stream_output(process, pdb_file, protein_id, prediction_id)

@app.route('/pdb/<path:filename>')
def serve_pdb(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
