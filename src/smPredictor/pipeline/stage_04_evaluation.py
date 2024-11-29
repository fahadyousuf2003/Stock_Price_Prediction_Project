from smPredictor.config.configuration import ConfigurationManager
from smPredictor.components.evaluation import Evaluation
from smPredictor import logger


STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(config=val_config)
        evaluation.create_rmse_directory()
        evaluation.evaluate()
        
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e