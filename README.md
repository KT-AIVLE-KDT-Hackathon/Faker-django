# Faker Django Backend

## Environment Setup

### Commands
```bash
# Create and activate environment for tensorflow
conda create -n <env_name> python=3.12
conda activate <env_name>

# Install model and django requirements
pip install -r requirements.txt
```

## Usage
```bash
python manage.py runserver
```

## REST API
`http://<host>:8000/convert` : Converts Image to Adversarial Image
- Method: POST
- Response: Blob of Adversarial Image
