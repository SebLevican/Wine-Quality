from src.datascience import logger
from src.datascience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascience.pipeline.data_validation_pipeline import DataValidationTraininigPipeline
from src.datascience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascience.pipeline.model_trainer_pipeline import ModelTrainderPipeline
from src.datascience.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline

logger.info('Welcome to the custom logging data science')


STAGE_NAME = 'Data Ingestion stage'


try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f'>>>>> stage {STAGE_NAME} completed\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Evaluation stage'

try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = DataValidationTraininigPipeline()
    data_ingestion.initiate_data_validation()
    logger.info(f'>>>>> stage {STAGE_NAME} completed\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Data Transformation stage'

try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = DataTransformationTrainingPipeline()
    data_ingestion.initiate_data_transformation()
    logger.info(f'>>>>> stage {STAGE_NAME} completed\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Trainer stage'

try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = ModelTrainderPipeline()
    data_ingestion.initiate_model_training()
    logger.info(f'>>>>> stage {STAGE_NAME} completed\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Evaluation stage'

try:
    logger.info(f'>>>> stage {STAGE_NAME} started <<<<<')
    data_ingestion = ModelEvaluationPipeline()
    data_ingestion.initiate_model_evaluation()
    logger.info(f'>>>>> stage {STAGE_NAME} completed\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e
