import base64
import os
from datetime import datetime
import streamlit as st
from main import *
import matplotlib.pyplot as plt


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Plot Knowledge Graph
def plot_KG(concepts):
    G = extract_KG(concepts)
    color_map = ['lightgreen' if node in concepts else 'yellow' for node in G.nodes()]
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.4)
    nx.draw(G, pos, node_color=color_map, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
    plt.title("Knowledge Graph")
    plt.axis("off")
    st.pyplot(plt)


# Main function
def main():
    set_png_as_page_bg('images/background.jpg')
    st.title("YOLO Object Detection and Knowledge Graph Visualization")

    use_webcam = st.checkbox("Use Webcam")

    uploaded_file = None
    image_path = None

    if not use_webcam:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
        if uploaded_file is not None:
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    else:
        picture = st.camera_input("Take a picture")
        if picture:
            temp_dir = "temp_images"
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, "webcam_image.png")
            with open(image_path, "wb") as f:
                f.write(picture.getbuffer())

    if image_path is not None:
        st.image(image_path, caption='Uploaded Image.', use_column_width=True)

        # Détection d'objets
        concepts = detect_objects(image_path)
        concepts = list(set(concepts))

        # Affichage des concepts dans un tableau avec fond noir
        st.markdown("""
            <style>
            table {
                width: 100%;
                background-color: black;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        st.table(concepts)

        # Vérification des concepts détectés
        if not concepts:
            st.error("Aucun objet détecté dans l'image. Veuillez essayer une autre image.")
            return

        # Dropdown pour sélectionner l'affichage
        option = st.selectbox(
            'Choose what to display:',
            ('Knowledge Graph', 'RDF Description')
        )

        if option == 'Knowledge Graph':
            rdf_graph = get_conceptnet_relations(concepts)
            insert_rdf_to_graphdb(rdf_graph)
            plot_KG(concepts)
        elif option == 'RDF Description':
            description = generate_rdf_description(concepts)
            st.text_area("RDF Description:", description , height=500)


if __name__ == '__main__':
    main()
