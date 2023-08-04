import spacy
from spacy import displacy

# Load the pre-trained named entity recognition model
NER = spacy.load("en_core_web_sm")

# Define a function for performing named entity recognition using the loaded model
def spacy_large_ner(document):
  # Use the NER model to extract named entities from the document
  # Return a set of tuples containing the extracted entities and their labels
  return {(ent.text.strip(), ent.label_) for ent in NER(document).ents}

# Define the document to be processed
doc = "The World Health Organization (WHO)[1] is a specialized agency of the United Nations responsible for international public health.[2] The WHO Constitution states its main objective as 'the attainment by all peoples of the highest possible level of health'.[3] Headquartered in Geneva, Switzerland, it has six regional offices and 150 field offices worldwide. The WHO was established on 7 April 1948.[4][5] The first meeting of the World Health Assembly (WHA), the agency's governing body, took place on 24 July of that year. The WHO incorporated the assets, personnel, and duties of the League of Nations' Health Organization and the Office International d'Hygiène Publique, including the International Classification of Diseases (ICD).[6] Its work began in earnest in 1951 after a significant infusion of financial and technical resources.[7]"

# Call the spacy_large_ner function to perform named entity recognition on the document
print(spacy_large_ner(doc))

# Render the named entities in the document using displacy and display the visualization
displacy.render(NER(doc), style="ent", jupyter=True)
