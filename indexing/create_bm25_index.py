from indexing.es_client import get_es_client
from indexing.indexer import create_index
from configs.loader import load_config

client = get_es_client()
config = load_config()
index_name = config['elasticsearch']['index_name']
create_index(client, index_name)
