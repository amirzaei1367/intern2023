from nltk.tag.stanford import StanfordNERTagger
jar = "stanford-ner-2020-11-17/stanford-ner-4.2.0.jar"
model = "stanford-ner-2020-11-17/classifiers/"
st_4class = StanfordNERTagger(model + "english.conll.4class.distsim.crf.ser.gz", jar, encoding='utf8')

doc="The World Health Organization (WHO) is a specialized agency of the United Nations responsible for international public health. Headquartered in Geneva, Switzerland, it has six regional offices and 150 field offices worldwide. The WHO was established on 7 April 1948. The first meeting of the World Health Assembly, the agency's governing body, took place on 24 July of that year. The WHO incorporated the assets, personnel, and duties of the League of Nations' Health Organization and the Office International d'Hygiène Publique, including the International Classification of Diseases (ICD)."
print(st_4class.tag(doc.split()))