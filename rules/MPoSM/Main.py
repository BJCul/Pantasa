import os
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

# Set AWS region and credentials as environment variables
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIATCKARXCFUKBIXCWE'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'nVPRxybdZ4Kkps9mZyOdrHCzsTalN38Be+eg5KST'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'  # Change to your desired AWS region

# Define the IAM role and S3 bucket for SageMaker
role = "arn:aws:iam::211125647499:role/MPoSMTrainer"
bucket = "pantasa-mposm"

def upload_to_s3(filename):
    try:
        s3 = boto3.client('s3')
        bucket = "pantasa-mposm"  # Use your existing bucket
        s3.upload_file(Filename=filename, Bucket=bucket, Key=f'tagalog-pos/{filename}')
        return f"s3://{bucket}/tagalog-pos/{filename}"
    except Exception as e:
        raise RuntimeError(f"Failed to upload file to S3: {str(e)}")

# Define the Hugging Face estimator for training
def fine_tune_model(train_file_s3):
    huggingface_estimator = HuggingFace(
        entry_point='Training.py',  # The training script
        source_dir='rules/MPoSM',  # The directory containing the training script
        instance_type='ml.m4.xlarge',
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
    try:
        train_file_s3 = upload_to_s3("rules/MPoSM/pos_tags_output.csv")
        print(f"Successfully uploaded file to {train_file_s3}")
    except Exception as e:
        print(f"Failed to upload file to S3: {str(e)}")
        return

    # Fine-tune the model
    try:
        fine_tune_model(train_file_s3)
        print("Training job started successfully.")
    except Exception as e:
        print(f"Failed to start the training job: {str(e)}")

if __name__ == "__main__":
    main()
