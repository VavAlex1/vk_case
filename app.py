import io
import streamlit as st
from PIL import Image
from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,

    Doc
)
import transformers
from transformers import AutoModel, AutoTokenizer
import torch


def ner(text):
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)
    ner_tagger = NewsNERTagger(emb)
    names_extractor = NamesExtractor(morph_vocab)

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)

    doc.tag_ner(ner_tagger)

    persons = set()
    organisations = set()
    for span in doc.spans:
        span.normalize(morph_vocab)
        if span.type == 'PER' and span.normal not in persons:
            persons.add(span.normal)
        elif span.type == 'ORG' and span.normal not in organisations:
            organisations.add(span.normal)

    if len(persons) > 0:
        st.write('**Личности:**')
        for name in persons:
            st.write(name)
    if len(organisations) > 0:
        st.write('**Организации:**')
        for organisation in organisations:
            st.write(organisation)


def load_text():
    uploaded_file = st.file_uploader(label='**Загрузите файл с текстом статьи:**')
    if uploaded_file is not None:
        text_data = uploaded_file.getvalue().decode("utf-8")
        st.write(text_data)
        ner(text_data)
        return text_data
    else:
        return None

st.title('Классификация текстов')
text = load_text()
