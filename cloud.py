# class that connects to gcp   

class Cloud():
    def __init__(self):
        self.project_id = 'mlops-project-220720'
        self.region = 'us-central1'
        self.zone = 'us-central1-a'
        self.bucket_name = 'mlops-bucket-220720'
        self.dataset_name = 'mlops-dataset-220720'
        self.model_name = 'mlops-model-220720'
        self.job_name = 'mlops-job-220720'
