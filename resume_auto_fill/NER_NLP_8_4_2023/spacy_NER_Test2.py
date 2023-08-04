import PyPDF2
import spacy
from spacy import displacy


#create file object variable
#opening method will be rb
pdffileobj=open('Casello_Joseph_RESUME.pdf','rb')
 
#create reader variable that will read the pdffileobj
pdfreader=PyPDF2.PdfReader(pdffileobj)
 
#This will store the number of pages of this pdf file
x=len(pdfreader.pages)
 
#create a variable that will select the selected number of pages
pageobj=pdfreader.pages[x-2]
 
#(x+1) because python indentation starts with 0.
#create text variable which will store all text datafrom pdf file
text=pageobj.extract_text()

# Load the pre-trained named entity recognition model
NER = spacy.load("en_core_web_lg")

# Define a function for performing named entity recognition using the loaded model
def spacy_large_ner(document):
  # Use the NER model to extract named entities from the document
  # Return a set of tuples containing the extracted entities and their labels
  return {(ent.text.strip(), ent.label_) for ent in NER(document).ents}

# Define the document to be processed
doc = text

# Call the spacy_large_ner function to perform named entity recognition on the document
print(spacy_large_ner(doc))

# Render the named entities in the document using displacy and display the visualization
displacy.render(NER(doc), style="ent", jupyter=True)
