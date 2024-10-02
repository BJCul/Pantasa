from sagemaker.huggingface import HuggingFace
import sagemaker

# Define the IAM role and S3 bucket for SageMaker
role = "your-sagemaker-execution-role"
bucket = "your-s3-bucket-name"

# Upload the processed dataset to S3
def upload_to_s3(filename):
    s3 = sagemaker.Session().default_bucket()
    s3.upload_data(path=filename, key_prefix='tagalog-pos')
    return f"s3://{bucket}/tagalog-pos/{filename}"

# Define the Hugging Face estimator for training
def fine_tune_model(train_file_s3):
    huggingface_estimator = HuggingFace(
        entry_point='Training.py',  # Training Script
        source_dir='preprocessed_output.csv',  # Directory containing any required source files
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        role=role,
        transformers_version='4.6',
        pytorch_version='1.7',
        py_version='py36',
        hyperparameters={
            'model_name_or_path': 'jcblaise/roberta-tagalog-base',
            'do_train': True,
            'train_file': train_file_s3,
            'output_dir': '/opt/ml/model',
            'learning_rate': 2e-5,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8
        }
    )

    # Start the training job
    huggingface_estimator.fit()

# Main function to execute the training process
def main():
    # Upload the processed CSV to S3
    train_file_s3 = upload_to_s3("processed_tagalog_data.csv")
    
    # Fine-tune the model
    fine_tune_model(train_file_s3)

if __name__ == "__main__":
    main()
