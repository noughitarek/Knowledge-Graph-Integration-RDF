from ultralytics import YOLO
import requests
from SPARQLWrapper import SPARQLWrapper, POST, TURTLE, JSON
import networkx as nx
import rdflib
import logging
from datetime import datetime


YOLO_MODEL = 'yolo11x.pt'
RDF_BASE_ENDPOINT = "http://localhost:7200"
REPOSITORY_NAME = "yolo"

# Object detection function
def detect_objects(image_path):
    logging.info(f"Starting object detection for image: {image_path}")
    try:
        model = YOLO('models/'+YOLO_MODEL)
        results = model(image_path, verbose=False)
        if not results or not results[0].boxes:
            logging.warning(f"No objects detected in image: {image_path}")
            return []
        classes = [model.names[int(box.cls)] for box in results[0].boxes]
        logging.info(f"Detected classes: {classes}")
        return classes
    except Exception as e:
        print(e)
        logging.error(f"Error during object detection: {e}")
        return []

# Fetch ConceptNet relations
def get_conceptnet_relations(concepts):
    """
    Fetches relations from ConceptNet for a list of given concepts and constructs an RDF graph.

    Args:
        concepts (list): A list of concepts (strings) for which ConceptNet relations will be fetched.

    Returns:
        rdflib.Graph: An RDF graph representing the relations between the concepts in ConceptNet.
    """
    
    base_url = "https://api.conceptnet.io/c/en/"
    graph = rdflib.Graph()

    logging.info("Starting to fetch ConceptNet relations for concepts.")
    CN = rdflib.Namespace("http://conceptnet.io")

    for concept in concepts:
        offset = 0
        limit = 2000
        logging.info(f"Fetching relations for concept: {concept}")

        while True:
            # Construct the URL to fetch relations for the current concept
            url = f"{base_url}{concept}?limit={limit}&offset={offset}"
            try:
                logging.debug(f"Requesting data from: {url}")
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for concept {concept} with error: {e}")
                break
            
            # Iterate over the edges (relations) in the fetched data
            for edge in data.get("edges", []):
                rel = edge.get("rel", {}).get("@id", "")
                start = edge.get("start", {}).get("@id", "")
                end = edge.get("end", {}).get("@id", "")

                # Skip if any required fields are missing
                if not rel or not start or not end:
                    continue
                
                # Create RDF URIs for the start concept, relation, and end concept
                start_uri = CN[start]
                end_uri = CN[end]
                rel_uri = CN[rel]

                # Add the relation to the RDF graph
                graph.add((start_uri, rel_uri, end_uri))
                logging.debug(f"Added relation: {start_uri} -> {rel_uri} -> {end_uri}")

            # Check if we have fetched all relations for the current concept
            if len(data.get("edges", [])) < limit:
                logging.info(f"Fetched all relations for concept: {concept}")
                break

            # Increment the offset to fetch the next set of relations
            offset += limit

    logging.info("Finished fetching ConceptNet relations.")
    return graph


# Insert RDF to GraphDB
def insert_rdf_to_graphdb(rdf_graph):
    """
    Inserts RDF data into a GraphDB repository.

    Args:
        rdf_graph (rdflib.Graph): The RDF graph to be inserted into GraphDB.
    
    Logs:
        - The progress of the insertion process.
        - Errors if any occur during the process.
    """
    logging.info("Starting RDF insertion to GraphDB.")

    try:
        # Initialize the SPARQLWrapper with the GraphDB repository endpoint
        sparql = SPARQLWrapper(RDF_BASE_ENDPOINT + "/repositories/" + REPOSITORY_NAME + "/statements")

        # Serialize the RDF graph to Turtle format for insertion
        rdf_data = rdf_graph.serialize(format='turtle')
        logging.debug(f"Serialized RDF data: {rdf_data[:200]}...")

        # Set up the SPARQL query to insert the RDF data into the GraphDB repository
        sparql.setQuery("""
            INSERT DATA { 
                %s
            }
        """ % rdf_data)

        # Set the request method and return format
        sparql.setMethod(POST)
        sparql.setReturnFormat(TURTLE)

        logging.info("Sending RDF data to GraphDB.")
        
        # Execute the query to insert the data
        sparql.query()
        logging.info("RDF data successfully inserted into GraphDB.")
    
    except Exception as e:
        logging.error(f"Error inserting RDF data into GraphDB: {e}")

# Extract Knowledge Graph
def extract_KG(concepts):
    # Build a SPARQL query that searches for relationships involving the provided concepts
    query = """
    PREFIX cn: <http://conceptnet.io/c/en/>
    SELECT ?subject ?predicate ?object
    WHERE {
        {
            ?subject ?predicate ?object .
            FILTER (?subject IN (""" + ",".join([f"cn:{concept}" for concept in concepts]) + """) ||
            ?object IN (""" + ",".join([f"cn:{concept}" for concept in concepts]) + """))
        }
        UNION
        {
            ?subject ?predicate ?object .
            ?object ?predicate2 ?subject2 .
            FILTER (?subject IN (""" + ",".join([f"cn:{concept}" for concept in concepts]) + """) ||
            ?object IN (""" + ",".join([f"cn:{concept}" for concept in concepts]) + """))
        }
    }
    LIMIT 1000
    """

    # Initialize the SPARQL wrapper and set the endpoint
    sparql = SPARQLWrapper(RDF_BASE_ENDPOINT+"/repositories/"+REPOSITORY_NAME)    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        # Execute the SPARQL query to retrieve data
        logging.debug("Sending SPARQL query to retrieve data.")
        results = sparql.query().convert()
        logging.info("SPARQL query executed successfully.")
    except Exception as e:
        # Log any errors during the query execution
        logging.error(f"Error executing SPARQL query: {e}")
        return
    
    # Initialize a new graph for the Knowledge Graph
    G = nx.Graph()
    logging.info("Building the graph from the query results.")

    # Iterate through the results and add edges to the graph
    for result in results["results"]["bindings"]:
        subject = result["subject"]["value"].split('/')[-1]
        predicate = result["predicate"]["value"].split('/')[-1]
        obj = result["object"]["value"].split('/')[-1]

        # Add an edge between the subject and object with the predicate as the label
        G.add_edge(subject, obj, label=predicate)

    logging.info("Graph construction complete. Now removing irrelevant nodes.")


    # Remove nodes that are not relevant to the given concepts
    nodes_to_remove = []

    for node in G.nodes():
        related_concepts = [n for n in G[node] if any(concept == n for concept in concepts)]
        if len(related_concepts) == 1 and not any(concept == node for concept in concepts):
            nodes_to_remove.append(node)

    G.remove_nodes_from(nodes_to_remove)
    
    # Return the constructed graph
    return G

# Generate RDF Description
def generate_rdf_description(concepts):
    graph = extract_KG(concepts)
    rdf_graph = rdflib.Graph()
    for u, v, data in graph.edges(data=True):
        subject = rdflib.URIRef(f"http://example.org/{u}")
        obj = rdflib.URIRef(f"http://example.org/{v}")
        predicate = rdflib.URIRef(f"http://example.org/{data['label']}")
        rdf_graph.add((subject, predicate, obj))
    return rdf_graph.serialize(format="turtle")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler('logs/'+datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log"), encoding='utf-8')
    ]
)