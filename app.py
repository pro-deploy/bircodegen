from flask import Flask, request, send_file, jsonify, render_template, g, make_response
import requests
import json
import os
import zipfile
import tempfile
from datetime import datetime
import sqlite3
import hashlib
from dotenv import load_dotenv
import traceback
import yaml

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Constants from env
HF_API_TOKEN = os.getenv('HF_API_TOKEN')
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/projects.db')
API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct/v1/chat/completions"

def detect_project_type(description):
    """Detect project type from description"""
    description_lower = description.lower()
    
    if any(x in description_lower for x in ['telegram', 'bot', 'tg bot']):
        return 'telegram_bot'
    elif any(x in description_lower for x in ['web', 'website', 'flask', 'django']):
        return 'web_app'
    elif any(x in description_lower for x in ['api', 'rest', 'graphql']):
        return 'api_service'
    elif any(x in description_lower for x in ['data', 'analytics', 'processing']):
        return 'data_processing'
    else:
        return 'general'

def get_project_requirements(project_type):
    """Get specific requirements based on project type"""
    requirements = {
        'telegram_bot': {
            'env_vars': [
                'TELEGRAM_BOT_TOKEN=your-telegram-bot-token',
                'BOT_NAME=your-bot-name',
                'WEBHOOK_URL=your-webhook-url (optional)',
            ],
            'dependencies': [
                'python-telegram-bot==20.7',
                'asyncio==3.4.3',
                'aiohttp==3.9.1',
            ],
            'services': {
                'redis': {
                    'image': 'redis:alpine',
                    'ports': ['6379:6379'],
                }
            }
        },
        'web_app': {
            'env_vars': [
                'DATABASE_URL=postgresql://user:password@db:5432/dbname',
                'SECRET_KEY=your-secret-key',
                'REDIS_URL=redis://redis:6379/0',
            ],
            'dependencies': [
                'flask==2.0.1',
                'sqlalchemy==1.4.23',
                'psycopg2-binary==2.9.1',
                'redis==4.1.0',
            ],
            'services': {
                'db': {
                    'image': 'postgres:13-alpine',
                    'environment': [
                        'POSTGRES_USER=user',
                        'POSTGRES_PASSWORD=password',
                        'POSTGRES_DB=dbname',
                    ],
                    'volumes': ['postgres_data:/var/lib/postgresql/data'],
                },
                'redis': {
                    'image': 'redis:alpine',
                    'ports': ['6379:6379'],
                }
            }
        },
        'api_service': {
            'env_vars': [
                'JWT_SECRET=your-jwt-secret',
                'API_KEY=your-api-key',
                'RATE_LIMIT=100',
            ],
            'dependencies': [
                'fastapi==0.70.0',
                'uvicorn==0.15.0',
                'pyjwt==2.3.0',
            ],
            'services': {}
        },
        'data_processing': {
            'env_vars': [
                'SPARK_MASTER=spark://spark-master:7077',
                'DATA_PATH=/data',
                'PROCESSING_INTERVAL=3600',
            ],
            'dependencies': [
                'pandas==1.3.3',
                'numpy==1.21.2',
                'scikit-learn==0.24.2',
            ],
            'services': {
                'spark-master': {
                    'image': 'bitnami/spark:latest',
                    'environment': ['SPARK_MODE=master'],
                    'ports': ['7077:7077', '8080:8080'],
                },
                'spark-worker': {
                    'image': 'bitnami/spark:latest',
                    'environment': [
                        'SPARK_MODE=worker',
                        'SPARK_MASTER_URL=spark://spark-master:7077',
                    ],
                }
            }
        },
        'general': {
            'env_vars': [
                'APP_ENV=development',
                'DEBUG=true',
                'LOG_LEVEL=INFO',
            ],
            'dependencies': [
                'requests==2.26.0',
                'python-dotenv==0.19.0',
            ],
            'services': {}
        }
    }
    return requirements.get(project_type, requirements['general'])

def parse_files_from_response(response_text, project_type, project_name="", project_description=""):
    """Parse files from model response and ensure all required files are present"""
    try:
        files = {}
        current_file = None
        current_content = []
        
        lines = response_text.split('\n')
        for line in lines:
            if line.startswith('```') and len(line) > 3:
                if current_file:
                    files[current_file] = '\n'.join(current_content).strip()
                    current_content = []
                current_file = line.replace('```', '').strip()
            elif line.startswith('```') and current_file:
                files[current_file] = '\n'.join(current_content).strip()
                current_file = None
                current_content = []
            elif current_file:
                current_content.append(line)

        # Final file check
        if current_file and current_content:
            files[current_file] = '\n'.join(current_content).strip()

        # Get project requirements
        requirements = get_project_requirements(project_type)
        
        # Ensure main application file exists based on project type
        main_files = {
            'telegram_bot': 'main.py',
            'web_app': 'app.py',
            'api_service': 'main.py',
            'data_processing': 'data_processor.py',
            'general': 'main.py'
        }
        main_filename = main_files.get(project_type, 'main.py')
        
        # Ensure all required files exist with proper extensions
        if not any(f.endswith('.py') for f in files):
            files[main_filename] = "# Main application code\n# Add your main application logic here"

        if 'requirements.txt' not in files:
            files['requirements.txt'] = '\n'.join(requirements['dependencies'])
        
        if '.env.example' not in files:
            files['.env.example'] = '\n'.join(requirements['env_vars'])
        
        if 'docker-compose.yml' not in files:
            services = requirements['services']
            docker_compose = {
                'version': '3.8',
                'services': {
                    'app': {
                        'build': '.',
                        'restart': 'unless-stopped',
                        'env_file': ['.env'],
                        'volumes': ['./data:/app/data'],
                        'ports': ['5000:5000']
                    },
                    **services
                }
            }
            if services:
                docker_compose['volumes'] = {'postgres_data': {}} if 'db' in services else {}
            files['docker-compose.yml'] = yaml.dump(docker_compose, default_flow_style=False)
        
        if 'Dockerfile' not in files:
            files['Dockerfile'] = f'''FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "{main_filename}"]'''
        
        if 'README.md' not in files:
            files['README.md'] = f'''# {project_name}

## Description
{project_description}

## Requirements
- Docker
- Docker Compose

## Setup
1. Clone the repository
2. Copy .env.example to .env and adjust variables
3. Run `docker-compose up --build`

## Project Structure
{chr(10).join(f"- {f}" for f in files.keys())}

## Development
[Development instructions]

## Production
[Production deployment instructions]'''
        
        if '.gitignore' not in files:
            files['.gitignore'] = '''# Python
__pycache__/
*.py[cod]
venv/
.env
.idea/
.vscode/
*.log
data/'''
        
        # Remove any empty files
        files = {k: v for k, v in files.items() if v.strip()}
        
        # Ensure all files have extensions
        for filename in list(files.keys()):
            if '.' not in filename:
                if 'main' in filename:
                    files[filename + '.py'] = files.pop(filename)
                elif 'requirements' in filename:
                    files[filename + '.txt'] = files.pop(filename)
                elif 'docker-compose' in filename:
                    files[filename + '.yml'] = files.pop(filename)
                elif 'Dockerfile' in filename:
                    continue  # Dockerfile doesn't need an extension
                else:
                    files[filename + '.txt'] = files.pop(filename)  # Default to .txt for unknown files

        return files
    except Exception as e:
        print(f"Error parsing files from response: {e}")
        traceback.print_exc()
        return {"main.py": response_text}

def get_db():
    try:
        db = getattr(g, '_database', None)
        if db is None:
            os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
            db = g._database = sqlite3.connect(DATABASE_PATH)
            db.row_factory = sqlite3.Row
        return db
    except Exception as e:
        print(f"Error getting database connection: {e}")
        traceback.print_exc()
        raise

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    try:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        c.executescript('''
            DROP TABLE IF EXISTS project_files;
            DROP TABLE IF EXISTS projects;
            
            CREATE TABLE projects
            (id TEXT PRIMARY KEY, 
             name TEXT NOT NULL,
             description TEXT NOT NULL,
             project_type TEXT NOT NULL,
             last_updated TIMESTAMP NOT NULL);
            
            CREATE TABLE project_files
            (project_id TEXT,
             filename TEXT,
             content TEXT,
             FOREIGN KEY(project_id) REFERENCES projects(id));
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        traceback.print_exc()
        raise

# Initialize DB at startup
with app.app_context():
    init_db()

def generate_project_id(description):
    return hashlib.md5(description.encode()).hexdigest()

def store_project(project_id, project_name, description, project_type, files):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        current_time = datetime.now().isoformat()
        c.execute('''INSERT OR REPLACE INTO projects 
                    (id, name, description, project_type, last_updated) 
                    VALUES (?, ?, ?, ?, ?)''', 
                 (project_id, project_name, description, project_type, current_time))
        
        c.execute("DELETE FROM project_files WHERE project_id = ?", (project_id,))
        
        for filename, content in files.items():
            c.execute("INSERT INTO project_files (project_id, filename, content) VALUES (?, ?, ?)",
                     (project_id, filename, content))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error storing project: {e}")
        traceback.print_exc()
        return False

def get_project(project_id):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        c.execute("SELECT name, description, project_type FROM projects WHERE id = ?", 
                 (project_id,))
        result = c.fetchone()
        
        if not result:
            conn.close()
            return None, None, None, None
            
        project_name, description, project_type = result
        
        c.execute("SELECT filename, content FROM project_files WHERE project_id = ?", 
                 (project_id,))
        files = {row[0]: row[1] for row in c.fetchall()}
        
        conn.close()
        return project_name, description, project_type, files
    except Exception as e:
        print(f"Error getting project: {e}")
        traceback.print_exc()
        return None, None, None, None

def query_model(messages, project_type):
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Get project-specific requirements
        requirements = get_project_requirements(project_type)
        
        # Update system message with project-specific requirements
        messages[0]["content"] = f"""You are a code generator. Generate complete project files based on description.
        This is a {project_type} project and must include these specific requirements:

        Environment Variables (.env.example must include):
        {chr(10).join(requirements['env_vars'])}

        Dependencies (requirements.txt must include):
        {chr(10).join(requirements['dependencies'])}

        Required Services:
        {json.dumps(requirements['services'], indent=2)}

        Generate ALL necessary files including:
        1. Main application code
        2. Dockerfile optimized for this project type
        3. docker-compose.yml with all required services
        4. .env.example with all required variables
        5. requirements.txt with all dependencies
        6. README.md with setup instructions
        7. .gitignore
        8. Any additional configuration files needed

        Use ```filename format for each file.
        Ensure all components work together in the containerized environment.
        Include health checks and proper error handling.
        Follow best practices for this type of project.
        """
        
        data = {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "messages": messages,
            "max_tokens": 4096,
            "stream": False
        }
        
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying model: {e}")
        traceback.print_exc()
        raise

def create_zip(files):
    try:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "project.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename, content in files.items():
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                zipf.write(file_path, filename)
        
        return zip_path
    except Exception as e:
        print(f"Error creating zip: {e}")
        traceback.print_exc()
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_code():
    try:
        if not request.json or 'description' not in request.json:
            return jsonify({"error": "No description provided"}), 400
            
        description = request.json['description']
        project_name = request.json.get('projectName', 'Unnamed Project')
        
        # Detect project type
        project_type = detect_project_type(description)
        project_id = generate_project_id(description)
        
        messages = [
            {
                "role": "system",
                "content": "You are a code generator. Generate complete project files based on description."
            },
            {
                "role": "user",
                "content": f"Generate a {project_type} project with following description: {description}"
            }
        ]
        
        try:
            response = query_model(messages, project_type)
            content = response['choices'][0]['message']['content']
            files = parse_files_from_response(content, project_type, project_name, description)
        except Exception as e:
            print(f"Error generating code: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to generate code"}), 500
        
        if not store_project(project_id, project_name, description, project_type, files):
            return jsonify({"error": "Failed to store project"}), 500
        
        try:
            zip_path = create_zip(files)
            response = make_response(send_file(zip_path, 
                                             as_attachment=True, 
                                             download_name='project.zip',
                                             mimetype='application/zip'))
            response.headers['X-Project-Id'] = project_id
            return response
        except Exception as e:
            print(f"Error creating zip file: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to create zip file"}), 500
            
    except Exception as e:
        print(f"Error in generate_code: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/update', methods=['POST'])
def update_code():
    try:
        if not request.json or 'project_id' not in request.json or 'new_feature' not in request.json:
            return jsonify({"error": "Missing required fields"}), 400
            
        project_id = request.json['project_id']
        new_feature = request.json['new_feature']
        
        project_name, description, project_type, existing_files = get_project(project_id)
        if not description:
            return jsonify({"error": "Project not found"}), 404
        
        files_context = "\n".join([f"File {f}:\n{c}" for f, c in existing_files.items()])
        
        messages = [
            {
                "role": "system",
                "content": "You are a code generator. Update existing project with new feature."
            },
            {
                "role": "user",
                "content": f"Existing {project_type} project:\n{files_context}\n\nAdd new feature: {new_feature}"
            }
        ]
        
        try:
            response = query_model(messages, project_type)
            content = response['choices'][0]['message']['content']
            updated_files = parse_files_from_response(
                content, 
                project_type,
                project_name, 
                f"{description}\n{new_feature}"
            )
        except Exception as e:
            print(f"Error updating code: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to update code"}), 500
        
        if not store_project(project_id, project_name, f"{description}\n{new_feature}", 
                           project_type, updated_files):
            return jsonify({"error": "Failed to store updated project"}), 500
        
        try:
            zip_path = create_zip(updated_files)
            response = make_response(send_file(zip_path, 
                                               as_attachment=True, 
                                               download_name='project.zip',
                                               mimetype='application/zip'))
            response.headers['X-Project-Id'] = project_id
            return response
        except Exception as e:
            print(f"Error creating zip file: {e}")
            traceback.print_exc()
            return jsonify({"error": "Failed to create zip file"}), 500
            
    except Exception as e:
        print(f"Error in update_code: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/projects', methods=['GET'])
def get_projects():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        c.execute('''SELECT id, name, description, project_type, last_updated 
                    FROM projects 
                    ORDER BY last_updated DESC''')
        
        projects = [{
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "project_type": row[3],
            "last_updated": row[4]
        } for row in c.fetchall()]
        
        conn.close()
        return jsonify(projects)
    except Exception as e:
        print(f"Error getting projects: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)