from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY,MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommenderPipeline:
    def __init__(self,persist_directory = "chroma_db"):
        try:
            logger.info("Initializing Recommendation Pipeline")
            vector_builder = VectorStoreBuilder(
                persist_directory=persist_directory,
                csv_path= "")
            retriever = vector_builder.load_vector_store().as_retriever()

            self.recommender = AnimeRecommender(retriever=retriever,
                                                model_name=MODEL_NAME,
                                                api_key=GROQ_API_KEY)
            logger.info("Recommendation Pipeline Initialized Successfully")
        except Exception as e:
            logger.error(f"Error initializing Recommendation Pipeline: {str(e)}")
            raise CustomException(f"Error initializing Recommendation Pipeline: {str(e)}")
    
    def recommend(self, query:str) -> str:
        try:
            logger.info(f"Generating recommendations for query: {query}")
            recommendations = self.recommender.get_recommendations(query)
            logger.info("Recommendations generated successfully")
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            raise CustomException(f"Error generating recommendations: {str(e)}")