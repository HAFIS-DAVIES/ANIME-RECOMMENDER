from src.vector_store import VectorStoreBuilder
from src.data_loader import AnimeDataLoader  
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException  

load_dotenv()

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting the Anime Recommender Pipeline")

        # Initialize Data Loader
        loader = AnimeDataLoader("data/anime_with_synopsis.csv", "data/anime_updates.csv")
        processed_csv = loader.load_and_process_data()
        logger.info(f"Data processed and saved to {processed_csv}")
        logger.info("Data loaded successfully")

        # Build Vector Store
        vector_builder = VectorStoreBuilder(
            persist_directory="chroma_db",
            csv_path=processed_csv
        )
        vector_builder.build_and_save_vector_store()
        logger.info("Vector store built and saved successfully")
        vector_builder = vector_builder.load_vector_store()
        logger.info("Vector store loaded successfully")
        
    except Exception as e:
        logger.error(f"An error occurred in the pipeline: {str(e)}")
        raise CustomException(f"An error occurred in the pipeline: {str(e)}")
    
if __name__ == "__main__":
    main()