#!/usr/bin/env python
# coding: utf-8

# # Hmw : Spark, File formats, and NLP
# 
# 
# 
# 
# 
# <p>Nous allons effectuer des analyses NLP sur des romans anglophones et francophones, que l'on regroupera dans un corpus romantique et un corpus réaliste (avec toute la pertinence de ces catégories). Les textes sont tout issus de la bibliothèque en ligne gutenberg.org
# 
# 

# In[1]:


import os
import sys
os.environ["JAVA_HOME"] = r"C:\Program Files\Microsoft\jdk-11.0.30.7-hotspot"  
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" +                      os.environ["HADOOP_HOME"] + r"\bin;" +                      os.environ["PATH"]

os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


# In[2]:


import pyspark
import sparknlp


# In[3]:


import plotly.io as pio
pio.renderers.default = "notebook"


# In[4]:


import urllib.request


# In[5]:


spark = sparknlp.start(gpu=True)


# In[6]:


# ============================================================
# PYSPARK
# ============================================================
from pyspark.sql import SparkSession
from pyspark.ml  import Pipeline
from pyspark.sql.functions import (
    col,
    lower,
    length,
    explode,
    posexplode,
    collect_list,
    concat_ws,
    avg,
    count,
    countDistinct,
    size,
    arrays_zip,
    udf,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    ArrayType,
)


from sparknlp.base import DocumentAssembler
from sparknlp.annotator import (
    SentenceDetector,
    Tokenizer,
    StopWordsCleaner,
    LemmatizerModel,
    PerceptronModel,        
    WordEmbeddingsModel,    
    NerDLModel,             
    NerConverter,
    SentenceDetectorDLModel,
)


import numpy  as np
import pandas as pd


import plotly.express       as px
import plotly.graph_objects as go
import plotly.io            as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"   


# In[7]:


import re


# In[8]:


from pyspark.sql.functions import lit
from functools import reduce


# In[9]:


dirr = "F:/books_tbd"
dir2 = "C:/Users/Utilisateur/Documents/analysestylo" #potentiellement à changer dans votre cas


# ## Chargement et prétraitement
# 
# Comme expliqué précédemment, les livres sont chargés en .txt depuis la plateforme Gutenberg. Certains romans sont en deux parties, il faut donc les charger à partir de deux ID (ou plus). L'on retire également le header de tous les textes de la plateforme Gutenberg (cela ne retire pas les éventuelles préfaces ou sommaires cela dit). On a un paramètre pour déterminer si l'on garde le fichier texte téléchargé ou non. Ce qui est renvoyé est un df d'une seule ligne.

# In[10]:


def load_gutenberg_book(book_id, title, save_dir=os.path.expanduser(dir2),keep_file=True):
    
    book_ids        = book_id if isinstance(book_id, list) else [book_id]
    all_clean_lines = []

    for bid in book_ids:
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_{bid}.txt")

        
        urls = [
            f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}.txt",
            f"https://www.gutenberg.org/files/{bid}/{bid}-8.txt",
        ]

        downloaded = False
        for url in urls:
            try:
                urllib.request.urlretrieve(url, save_path)
                downloaded = True
                break
            except Exception:
                continue

        if not downloaded:
            print(f"  WARNING: Could not download book {bid}")
            continue

        
        lines = None
        for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
            try:
                with open(save_path, "r", encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue

        if lines is None:
            print(f"  WARNING: Could not decode book {bid}")
            continue

        try:
            start = next(i for i, l in enumerate(lines) if "START OF" in l and "GUTENBERG" in l)
            end   = next(i for i, l in enumerate(lines) if "END OF"   in l and "GUTENBERG" in l)
            all_clean_lines.extend(lines[start+1:end])
        except StopIteration:
            all_clean_lines.extend(lines)

    clean_df  = spark.createDataFrame(
        [(line.strip(),) for line in all_clean_lines if line.strip()],
        ["text"]
    )
    

    if not keep_file:
        for bid in book_ids:
            save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_{bid}.txt")
            try:
                os.remove(save_path)
            except FileNotFoundError:
                pass

    return clean_df.select(concat_ws(" ", collect_list("text")).alias("text"))

def load_book_with_label(book_id, title, genre, save_dir=os.path.expanduser(dir2),keep_file=True): #2e version, charger avec "Romantique" ou "Réaliste"
    

    book_ids        = book_id if isinstance(book_id, list) else [book_id]
    all_clean_lines = []

    for bid in book_ids:
        url       = f"https://www.gutenberg.org/files/{bid}/{bid}-0.txt"
        save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_{bid}.txt")

        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception:
            url = f"https://www.gutenberg.org/files/{bid}/{bid}.txt"
            urllib.request.urlretrieve(url, save_path)

        with open(save_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        try:
            start = next(i for i, l in enumerate(lines) if "START OF" in l and "GUTENBERG" in l)
            end   = next(i for i, l in enumerate(lines) if "END OF"   in l and "GUTENBERG" in l)
            all_clean_lines.extend(lines[start+1:end])
        except StopIteration:
            all_clean_lines.extend(lines)

    df = spark.createDataFrame(
        [(line.strip(),) for line in all_clean_lines if line.strip()],
        ["text"]
    )

    return df.withColumn("title", lit(title))              .withColumn("genre", lit(genre))


    if not keep_file:
            for bid in book_ids:
                save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_{bid}.txt")
                try:
                    os.remove(save_path)
                except FileNotFoundError:
                    pass



# ## Pipeline 
# 
# DocumentAssembler : Transforme le texte en document NLP.
# 
# SentenceDetector : Sépare les documents en phrases.
# 
# Tokenizer : Sépare les phrases en mots/tokens.
# 
# StopWordsCleaner : Retire les mots du comme des déterminants ("the","le") conjonctions ("and","et") etc. (cette liste de mots est définie manuellement)
# 
# Pos_tagger : Assigne des types aux mots (nom commun, verbe etc.)
# 
# Les pipelines anglophones et francophones sont différentes mais ont la même structure.

# In[11]:



# PIPELINE ANGLOPHONE
stopwords = [
    "the", "a", "an", "and", "of", "to", "in", "was", "is", "are",
    "he", "she", "it", "that", "her", "his", "had", "i", "be", "as",
    "at", "by", "we", "or", "but", "not", "with", "for", "on", "so",
    "my", "you", "have", "from", "they", "all", "me", "no", "do", "if",
    "been", "would", "could", "should", "than", "then", "them", "their",
    "what", "which", "who", "this", "were", "has", "one", "when", "up",
    "said", "mr", "mrs", "miss", "s", "t", "upon", "into", "out", "about",
    "there", "will", "very", "much", "more", "own", "must", "such", "now",
    "every", "never", "being", "any", "am", "its", "our", "your", "also"
]





document_assembler = DocumentAssembler()     .setInputCol("text")     .setOutputCol("document")

sentence_detector = SentenceDetector()     .setInputCols(["document"])     .setOutputCol("sentence")

tokenizer = Tokenizer()     .setInputCols(["sentence"])     .setOutputCol("token")

stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_en", "en")     .setInputCols(["token"])     .setOutputCol("clean_token")     .setCaseSensitive(False)

pos_tagger = PerceptronModel.pretrained("pos_anc", "en")     .setInputCols(["sentence", "clean_token"])     .setOutputCol("pos")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    stopwords_cleaner,
    pos_tagger
])


### Pipeline francophone 

stopwords_fr = [
    "le", "la", "les", "un", "une", "des", "du", "de", "d", "l",
    "et", "en", "au", "aux", "ce", "qui", "que", "qu", "se", "sa",
    "son", "ses", "mon", "ma", "mes", "ton", "ta", "tes", "je", "tu",
    "il", "elle", "nous", "vous", "ils", "elles", "on", "y", "ne",
    "pas", "plus", "par", "sur", "dans", "avec", "est", "sont", "était",
    "être", "avoir", "été", "fait", "faire", "dit", "dire", "tout",
    "mais", "ou", "donc", "or", "ni", "car", "si", "car", "comme",
    "bien", "aussi", "même", "alors", "encore", "toujours", "très",
    "lui", "leur", "leurs", "cet", "cette", "ces", "dont", "où"
]








document_assembler_fr = DocumentAssembler()     .setInputCol("text")     .setOutputCol("document")

sentence_detector_fr = SentenceDetector()     .setInputCols(["document"])     .setOutputCol("sentence")


tokenizer_fr = Tokenizer()     .setInputCols(["sentence"])     .setOutputCol("token")

stopwords_cleaner_fr = StopWordsCleaner.pretrained("stopwords_fr", "fr")     .setInputCols(["token"])     .setOutputCol("clean_token")     .setCaseSensitive(False)

pos_tagger_fr = PerceptronModel.pretrained("pos_ud_gsd", "fr")     .setInputCols(["sentence", "clean_token"])     .setOutputCol("pos")

pipeline_fr = Pipeline(stages=[
    document_assembler_fr,
    sentence_detector_fr,
    tokenizer_fr,
    stopwords_cleaner_fr,
    pos_tagger_fr
])


# ## Analyse stylométrique
# 
# On va principalement calculer les Flesh-Kincaid. Cela nécessite de créer des compteurs de syllabes pour les mots (qui dépendent de la langue). Plus le Flesh-Kincaid est élevé, plus le texte est simple. À l'inverse, le FK-grade est censé estimer la classe minimale d'un élève pouvant lire un texte : plus in est élevé, plus le texte est compliqué (à titre indicatif, 8 est l'équivalent de la quatrième en France). L'on séparera aussi les textes en dialogue et en narration (en utilisant les guillemets et les verbes de dialogue). Kandel-Moles est un autre test de lisibilité, apparemment plus adapté au français.
# 
# L'on vérifiera également la loi de Zipf, qui établit une relation entre l'ordre des mots selon leur fréquence et leur fréquence.

# In[13]:



def count_syllables(word):
    
    word = word.lower().strip(".:;?!")
    if len(word) == 0:
        return 0
    vowels    = "aeiouy"
    count     = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def compute_readability_windows(sentences, window_sizes=[10, 25, 50, 100],
                                syllable_fn=count_syllables):
    results  = {}
    sentences = [s for s in sentences if len(s.split()) < 500]

    for ws in window_sizes:
        records = []
        for i in range(0, len(sentences) - ws, ws // 2):
            window = sentences[i:i + ws]
            words  = " ".join(window).split()
            n_sent = len(window)
            n_word = len(words)
            if n_word == 0 or n_sent == 0:
                continue
            n_syl  = sum(syllable_fn(w) for w in words) 
            asl    = n_word / n_sent
            asw    = n_syl  / n_word
            records.append({
                "window_start": i,
                "window_mid":   i + ws // 2,
                "flesch":       206.835 - (1.015 * asl) - (84.6  * asw),
                "fk_grade":     (0.39   * asl) + (11.8  * asw) - 15.59,
                "kandel_moles": 209.835 - (1.015 * asl) - (84.6  * asw)
            })
        results[ws] = pd.DataFrame(records)
    return results


def readability_for_group(group_df): 
    
    n_sent = len(group_df)
    n_word = group_df["length"].sum()
    n_syl  = group_df["syllables"].sum()

    if n_word == 0 or n_sent == 0:
        return {}

    asl = n_word / n_sent
    asw = n_syl  / n_word

    return {
        "count":        n_sent,
        "avg_length":   asl,
        "flesch":       206.835 - (1.015 * asl) - (84.6  * asw),
        "fk_grade":     (0.39   * asl) + (11.8  * asw) - 15.59,
        "kandel_moles": 209.835 - (1.015 * asl) - (84.6  * asw)
    }


def segment_text(sentences, lang="en"): #segmente en dialogue VS narration
    

    if lang == "fr":
        speech_verbs = r'\b(dit|répondit|demanda|s\'écria|s\'exclama|murmura|déclara|ajouta|continua|reprit|souffla|cria)\b'
        dialog_start = r'^[«"\u201c]'
        dialog_end   = r'[»"\u201d]$'
        dialog_dash  = r'^\s*[-—]'
    else:
        speech_verbs = r'\b(said|replied|asked|cried|exclaimed|answered|whispered|shouted|murmured|declared|added|continued)\b'
        dialog_start = r'^["\u201c\u2018]'
        dialog_end   = r'["\u201d\u2019]$'
        dialog_dash  = None

    records = []
    for i, sent in enumerate(sentences):
        s         = sent.strip()
        is_dialog = (
            bool(re.match(dialog_start, s)) or
            bool(re.search(dialog_end, s)) or
            bool(re.search(speech_verbs, s, re.IGNORECASE)) or
            (dialog_dash is not None and bool(re.search(dialog_dash, s)))
        )
        records.append({
            "idx":       i,
            "sentence":  s,
            "type":      "Dialog" if is_dialog else "Narration",
            "length":    len(s.split()),
            "syllables": sum(count_syllables(w) for w in s.split())
        })

    return pd.DataFrame(records)


def _plot_readability(window_results, title, save_dir=None,display=False):
    fig    = make_subplots(rows=3, cols=1,
                subplot_titles=("Flesch Reading Ease",
                                "Flesch-Kincaid Grade Level",
                                "Kandel-Moles"),
                shared_xaxes=True)
    colors = {10: "#e74c3c", 25: "#e67e22", 50: "#2980b9", 100: "#27ae60"}

    for ws, df_w in window_results.items():
        for row, metric in enumerate(["flesch", "fk_grade", "kandel_moles"], start=1):
            fig.add_trace(go.Scatter(
                x=df_w["window_mid"], y=df_w[metric],
                mode="lines", name=f"Window={ws}",
                line=dict(color=colors[ws], width=1.5),
                legendgroup=f"ws{ws}",
                showlegend=(row == 1)
            ), row=row, col=1)

    fig.add_hline(y=60, line_dash="dash", line_color="gray",
                  annotation_text="Standard (60)", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot",  line_color="red",
                  annotation_text="Very Hard (30)", row=1, col=1)
    fig.update_layout(height=900,
                      title_text=f"Readability Stability — {title}",
                      template="plotly_white")
    
    
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"{title}_readability.html"))
        print(f"  Saved: {title}_readability.html")

    if display:
        fig.show()
    

def _plot_zipf(word_freq_pd, title, save_dir=None,display=False):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=word_freq_pd["rank"], y=word_freq_pd["count"],
        mode="lines", name="Observed",
        line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(
        x=word_freq_pd["rank"], y=word_freq_pd["zipf_ideal"],
        mode="lines", name="Ideal Zipf (1/rank)",
        line=dict(color="#2980b9", width=2, dash="dash")))
    fig.update_layout(
        title=f"Zipf Plot — {title}",
        xaxis=dict(title="Rank",      type="log"),
        yaxis=dict(title="Frequency", type="log"),
        template="plotly_white")
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"{title}_zipf.html"))
        print(f"  Saved: {title}_zipf.html")

    if display:
        fig.show()
    
    



def _plot_word_freq(word_freq_pd, title, save_dir=None,display=False):
    top30 = word_freq_pd         .sort_values("count", ascending=False)         .head(30)
    fig   = px.bar(top30, x="word", y="count",
                   title=f"Top 30 Words — {title}",
                   template="plotly_white",
                   color_discrete_sequence=["#e07b39"])
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"{title}_wordfreq.html"))
        print(f"  Saved: {title}_wordfreq.html")

    if display:
        fig.show()

def _plot_segments(segments_df, dialog_stats, narration_stats, title, save_dir=None,display=False):
    
    fig = make_subplots(rows=1, cols=3,
            subplot_titles=("Sentence Count",
                            "Avg Sentence Length",
                            "Flesch Reading Ease"),
            specs=[[{"type":"xy"}, {"type":"xy"}, {"type":"xy"}]])

    for col_idx, metric in enumerate(["count", "avg_length", "flesch"], start=1):
        fig.add_trace(go.Bar(
            x=["Dialog", "Narration"],
            y=[dialog_stats.get(metric, 0), narration_stats.get(metric, 0)],
            marker_color=["#e74c3c", "#2980b9"],
            showlegend=False
        ), row=1, col=col_idx)

    fig.update_layout(title_text=f"Dialog vs Narration — {title}",
                      template="plotly_white")
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"{title}_dialog_narration.html"))
        print(f"  Saved: {title}_dialog_narration.html")

    if display:
        fig.show()

    
    segments_df["dialog_ratio"] = (
        (segments_df["type"] == "Dialog")
        .astype(int)
        .rolling(50).mean()
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=segments_df["idx"], y=segments_df["dialog_ratio"],
        mode="lines", fill="tozeroy",
        line=dict(color="#e74c3c"), name="Dialog Ratio"))
    fig2.update_layout(
        title=f"Dialog Proportion Across Novel — {title}",
        xaxis_title="Sentence Index",
        yaxis_title="Proportion of Dialog",
        yaxis=dict(range=[0, 1]),
        template="plotly_white")
    if save_dir:
        fig2.write_html(os.path.join(save_dir, f"{title}_dialog_ratio.html"))
        print(f"  Saved: {title}_dialog_ratio.html")

    if display:
        fig2.show()


# In[19]:




def analyze_book(full_text, title, save_dir=None,display=False): #analyse stylométrique complète d'un livre
    

    
    result = pipeline.fit(full_text).transform(full_text)

    
    sentences_pd = result.select(
        explode(col("sentence")).alias("sent")
    ).select(col("sent.result").alias("sentence")).toPandas()

    tokens_pd = result.select(
        explode(col("token.result")).alias("word")
    ).select(lower(col("word")).alias("word")) \
     .filter(col("word").rlike("^[a-zA-Z]{2,}$")) \
     .toPandas()

    tokens_pd["syllables"] = tokens_pd["word"].apply(count_syllables)

    
    num_sentences = len(sentences_pd)
    num_words     = len(tokens_pd)
    num_syllables = tokens_pd["syllables"].sum()
    asl           = num_words     / num_sentences
    asw           = num_syllables / num_words
    flesch        = 206.835 - (1.015 * asl) - (84.6 * asw)
    fk_grade      = (0.39   * asl) + (11.8  * asw) - 15.59
    kandel        = 209.835 - (1.015 * asl) - (84.6 * asw)

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Sentences:            {num_sentences}")
    print(f"  Words:                {num_words}")
    print(f"  Avg sentence length:  {asl:.2f}")
    print(f"  Avg syllables/word:   {asw:.2f}")
    print(f"  Flesch Reading Ease:  {flesch:.2f}")
    print(f"  FK Grade Level:       {fk_grade:.2f}")
    print(f"  Kandel-Moles:         {kandel:.2f}")

    # lisibilité par fenêtre glissante
    sentences_list = sentences_pd["sentence"].tolist()
    window_results = compute_readability_windows(sentences_list, [10, 25, 50, 100])

    #fréquence des mots
    word_freq_pd = tokens_pd[~tokens_pd["word"].isin(stopwords)]         .groupby("word")["word"].count()         .reset_index(name="count")         .sort_values("count", ascending=False)

    # zipf
    word_freq_pd["rank"]       = range(1, len(word_freq_pd) + 1)
    word_freq_pd["zipf_ideal"] = word_freq_pd["count"].max() / word_freq_pd["rank"]

    
    segments_df = segment_text(sentences_list)
    dialog_stats    = readability_for_group(segments_df[segments_df["type"] == "Dialog"])
    narration_stats = readability_for_group(segments_df[segments_df["type"] == "Narration"])

    
    _plot_readability(window_results, title, save_dir,display)
    _plot_zipf(word_freq_pd, title, save_dir,display)
    _plot_segments(segments_df, dialog_stats, narration_stats, title, save_dir,display)
    _plot_word_freq(word_freq_pd, title, save_dir,display)

    return {
        "title":        title,
        "n_sentences":  num_sentences,
        "n_words":      num_words,
        "asl":          asl,
        "asw":          asw,
        "flesch":       flesch,
        "fk_grade":     fk_grade,
        "kandel":       kandel,
        "word_freq":    word_freq_pd,
        "segments":     segments_df,
        "windows":      window_results,
        "sentences":    sentences_pd
    }


def count_syllables_fr(word):
    
    word = word.lower().strip(".:;?!«»,;")
    if len(word) == 0:
        return 0

    
    simple_vowels  = "aeiouyàâáéèêùûôîœæ"
    
    diaeresis      = "äëïüÿ"

    
    digraphs = [
        "eau", "oeu", "oè",           
        "ou", "ai", "ei", "au", "eu",
        "oi", "ui", "oe", "ae", "œu"
    ]

    count = 0
    i     = 0

    while i < len(word):
        char = word[i]

        
        if char in diaeresis:
            count += 1
            i     += 1
            continue

        
        matched = False
        for digraph in digraphs:
            end = i + len(digraph)
            if word[i:end] == digraph:
                
                if end < len(word) and word[end] in diaeresis:
                    break  
                count  += 1
                i       = end
                matched = True
                break

        if matched:
            continue

       
        if char in simple_vowels:
            count += 1

        i += 1

    
    if (word.endswith("e") and
        not word.endswith("ée") and
        not word.endswith("ie") and
        count > 1):
        count -= 1

   
    if word.endswith("es") and count > 1:
        count -= 1
    if word.endswith("ent") and count > 1:
        count -= 1

    return max(1, count)

def analyze_book_fr(full_text, title, save_dir=os.path.expanduser(dir2)):
    

    result = pipeline_fr.fit(full_text).transform(full_text)

    sentences_pd = result.select(
        explode(col("sentence")).alias("sent")
    ).select(col("sent.result").alias("sentence")).toPandas()

    tokens_pd = result.select(
        explode(col("token.result")).alias("word")
    ).select(lower(col("word")).alias("word")) \
     .filter(col("word").rlike("^[a-zA-Zàâäéèêëîïôùûüœæ]{2,}$")) \
     .toPandas()

    tokens_pd["syllables"] = tokens_pd["word"].apply(count_syllables_fr)

    n_sent = len(sentences_pd)
    n_word = len(tokens_pd)
    n_syl  = tokens_pd["syllables"].sum()
    asl    = n_word / n_sent
    asw    = n_syl  / n_word

    flesch = 206.835 - (1.015 * asl) - (84.6  * asw)
    fk     = (0.39   * asl) + (11.8  * asw) - 15.59
    kandel = 209.835 - (1.015 * asl) - (84.6  * asw)

    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  Sentences:            {n_sent}")
    print(f"  Words:                {n_word}")
    print(f"  Avg sentence length:  {asl:.2f}")
    print(f"  Avg syllables/word:   {asw:.2f}")
    print(f"  Flesch Reading Ease:  {flesch:.2f}")
    print(f"  FK Grade Level:       {fk:.2f}")
    print(f"  Kandel-Moles:         {kandel:.2f}")

    sentences_list = sentences_pd["sentence"].tolist()
    sentences_list = [s for s in sentences_list if len(s.split()) < 500]
    window_results = compute_readability_windows(
        sentences_list, [10, 25, 50, 100],
        syllable_fn=count_syllables_fr   
    )

    word_freq_pd = tokens_pd[~tokens_pd["word"].isin(stopwords_fr)]         .groupby("word")["word"].count()         .reset_index(name="count")         .sort_values("count", ascending=False)

    word_freq_pd["rank"]       = range(1, len(word_freq_pd) + 1)
    word_freq_pd["zipf_ideal"] = word_freq_pd["count"].max() / word_freq_pd["rank"]

    segments_df     = segment_text(sentences_list,lang="fr")
    dialog_stats    = readability_for_group(segments_df[segments_df["type"] == "Dialog"])
    narration_stats = readability_for_group(segments_df[segments_df["type"] == "Narration"])

    _plot_readability(window_results, title, save_dir)
    _plot_zipf(word_freq_pd, title, save_dir)
    _plot_segments(segments_df, dialog_stats, narration_stats, title, save_dir)
    _plot_word_freq(word_freq_pd, title, save_dir)

    return {
        "title":        title,
        "genre":        None,   
        "n_sentences":  n_sent,
        "n_words":      n_word,
        "asl":          asl,
        "asw":          asw,
        "flesch":       flesch,
        "fk_grade":     fk,
        "kandel":       kandel,
        "word_freq":    word_freq_pd,
        "segments":     segments_df,
        "sentences":    sentences_pd,
        "windows":      window_results
    }


def analyze_corpus_books_fr(book_list, genre, save_dir=os.path.expanduser(dir2)):
    
    all_stats = []
    for book_id, title in book_list:
        print(f"\nProcessing: {title} ({genre})")
        try:
            full_text      = load_gutenberg_book(book_id, title, save_dir)
            stats          = analyze_book_fr(full_text, title, save_dir)
            stats["genre"] = genre
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR on {title}: {e}")
            import traceback
            traceback.print_exc()
    return all_stats


# ### Exemple sur deux livres 
# 
# On effectue les analyses stylométriques sur Moby Dick de Melville et Ulysses de James Joyce (qui n'étant ni romantique ni réaliste, ne réapparaître malheureusement pas).

# In[15]:


ulysses = load_gutenberg_book(4300,"Ulysses")
ulysses_stats = analyze_book(ulysses,"Ulysses",display=True)


# L'on remarque une chute de la lisibilité vers la fin du livre (à vrai dire, c'était pire avant de retirer l'outlier de la dernière partie du livre qui est un soliloque très très long), ainsi qu'une partie narrative plus lisible que les dialogues (c'est en général l'inverse). Curieusement, d'après les indicateurs de Flesh-Kincaid, le livre est relativement lisible (un élève en début de collège serait capable de le lire ?).

# In[17]:


mobydick = load_gutenberg_book(2701,"Moby Dick")
mobydick_stats = analyze_book(mobydick, "Moby Dick",display=True)


# Le mot "whale" est le plus commun (attendu ?). Il serait aussi plus difficile à lire que Ulysses. Ce qui est vraiment derrière est que l'indicateur Flesh-Kincaid ne se base que sur la longueur des mots, phrases et la variété du vocabulaire, ce qui n'est pas là vraiment que repose la difficulté de certains textes (par exemple Ulysses).

# ## Analyse par corpus
# 
# L'on va maintenant analyser les livres par corpus. Les données seront stockées dans un fichier parquet.

# In[20]:





def load_corpus(book_list, genre, save_dir=os.path.expanduser(dir2)):
    
    dfs = [
        load_book_with_label(book_id, title, genre, save_dir)
        for book_id, title in book_list
    ]
    # Union all books into a single DataFrame
    return reduce(lambda a, b: a.union(b), dfs)


def analyze_corpus_books(book_list, genre, save_dir=os.path.expanduser(dir2)): #renvoie une liste de dictionnaires
    
    all_stats = []

    for book_id, title in book_list:
        print(f"\nProcessing: {title} ({genre})")
        try:
            full_text  = load_gutenberg_book(book_id, title, save_dir)
            stats      = analyze_book(full_text, title, save_dir)
            stats["genre"] = genre   # tag with genre
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR on {title}: {e}")

    return all_stats


def plot_books_comparison(all_stats, save_dir=None,display=False,lang="en"):

    df = pd.DataFrame([{
        "title":  s["title"],
        "genre":  s["genre"],
        "flesch": s["flesch"],
        "fk":     s["fk_grade"],
        "kandel": s["kandel"],
        "asl":    s["asl"],
        "asw":    s["asw"],
        "n_words":s["n_words"],
    } for s in all_stats])

    color_map = {"Romantique": "#e74c3c", "Réaliste": "#2980b9"}
    metrics   = ["flesch", "fk", "kandel", "asl", "asw", "n_words"]
    titles    = ["Flesch Reading Ease", "FK Grade Level", "Kandel-Moles",
                 "Avg Sentence Length", "Avg Syllables/Word", "Total Words"]

    fig = make_subplots(rows=2, cols=3, subplot_titles=titles)

    for idx, metric in enumerate(metrics):
        row = idx // 3 + 1
        col = idx  %  3 + 1

        for genre, color in color_map.items():
            subset = df[df["genre"] == genre]
            fig.add_trace(go.Bar(
                x=subset["title"],
                y=subset[metric],
                name=genre,
                marker_color=color,
                showlegend=(idx == 0)
            ), row=row, col=col)

    fig.update_layout(
        height=700,
        title_text="Individual Novel Comparison — Romantique vs Réaliste",
        template="plotly_white",
        barmode="group"
    )
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"books_comparison_{lang}.html"))
        print(f"books_comparison_{lang}.html")

    if display:
        fig.show()


def analyze_full_corpus_from_stats(stats_list, genre): #fait des statistiques par corpus, à partir des statistiques par livre
    
    total_sentences = sum(s["n_sentences"] for s in stats_list)
    total_words     = sum(s["n_words"]     for s in stats_list)

    
    def weighted_avg(metric):
        return sum(s[metric] * s["n_words"] for s in stats_list) / total_words

    asl    = weighted_avg("asl")
    asw    = weighted_avg("asw")
    flesch = 206.835 - (1.015 * asl) - (84.6  * asw)
    fk     = (0.39   * asl) + (11.8  * asw) - 15.59
    kandel = 209.835 - (1.015 * asl) - (84.6  * asw)

    
    all_word_freqs = pd.concat(
        [s["word_freq"] for s in stats_list],
        ignore_index=True
    ).groupby("word")["count"].sum() \
     .reset_index() \
     .sort_values("count", ascending=False)

    all_word_freqs["rank"]       = range(1, len(all_word_freqs) + 1)
    all_word_freqs["zipf_ideal"] = all_word_freqs["count"].max() / all_word_freqs["rank"]

    
    all_segments = pd.concat(
        [s["segments"] for s in stats_list],
        ignore_index=True
    )
    dialog_stats    = readability_for_group(all_segments[all_segments["type"] == "Dialog"])
    narration_stats = readability_for_group(all_segments[all_segments["type"] == "Narration"])

    return {
        "genre":        genre,
        "n_sentences":  total_sentences,
        "n_words":      total_words,
        "asl":          asl,
        "asw":          asw,
        "flesch":       flesch,
        "fk_grade":     fk,
        "kandel":       kandel,
        "ttr":          sum(s["n_words"] for s in stats_list),  
        "word_freq":    all_word_freqs,
        "segments":     all_segments,
        "sentences":    pd.concat([s.get("sentences", pd.DataFrame()) 
                                   for s in stats_list], ignore_index=True)
    }





    
    
def plot_corpus_comparison(romantic_corpus, realist_corpus, save_dir=None,display=False,lang="en"): #comparaison des deux corpus (corpora ?)
    

    corpora = [romantic_corpus, realist_corpus]
    labels  = [c["genre"] for c in corpora]
    colors  = ["#e74c3c", "#2980b9"]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Flesch Ease", "FK Grade", "Avg Sent Length", "TTR"),
        specs=[[{"type":"xy"}]*4]
    )

    for col_idx, metric in enumerate(["flesch", "fk_grade", "asl", "ttr"], start=1):
        fig.add_trace(go.Bar(
            x=labels,
            y=[c[metric] for c in corpora],
            marker_color=colors,
            showlegend=False
        ), row=1, col=col_idx)

    fig.update_layout(
        title_text="Corpus-Level Comparison — Romantic vs Realist",
        template="plotly_white",
        height=400
    )
    

    if display:
        fig.show()
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"corpus_comparison_{lang}.html"))
    
    

    
    fig_zipf = go.Figure()
    colors_zipf = {"Romantique": "#e74c3c", "Réaliste": "#2980b9"}

    for corpus in [romantic_corpus, realist_corpus]:
        sentences_list = corpus["sentences"]["sentence"].tolist()
        sentences_list = [s for s in sentences_list if len(s.split()) < 500]

        tokens = " ".join(sentences_list).split()
        freq   = pd.Series(tokens).value_counts().reset_index()
        freq.columns = ["word", "count"]
        freq   = freq[~freq["word"].isin(stopwords)]
        freq["rank"]       = range(1, len(freq) + 1)
        freq["zipf_ideal"] = freq["count"].max() / freq["rank"]

        fig_zipf.add_trace(go.Scatter(
            x=freq["rank"], y=freq["count"],
            mode="lines",
            name=f"{corpus['genre']} (observed)",
            line=dict(color=colors_zipf[corpus["genre"]], width=2)
        ))
        fig_zipf.add_trace(go.Scatter(
            x=freq["rank"], y=freq["zipf_ideal"],
            mode="lines",
            name=f"{corpus['genre']} (ideal Zipf)",
            line=dict(color=colors_zipf[corpus["genre"]], width=1, dash="dash")
        ))

    fig_zipf.update_layout(
        title="Zipf Comparison — Romantic vs Realist Corpus",
        xaxis=dict(title="Rank",      type="log"),
        yaxis=dict(title="Frequency", type="log"),
        template="plotly_white"
    )
    
    
    
    
    
    
    

    if display:
        fig_zipf.show()
    if save_dir:
        fig.write_html(os.path.join(save_dir, f"corpus_zipf_{lang}.html"))
        print(f"Saved: corpus_zipf_{lang}.html")

def save_results_parquet(all_stats, romantic_corpus, realist_corpus,
                         lang="en", save_dir=os.path.expanduser(dir2)):
    parquet_dir = os.path.join(save_dir, "parquet_results")
    suffix      = f"_{lang}"

    
    summary_df = spark.createDataFrame(pd.DataFrame([{
        "title":       s["title"],
        "genre":       s["genre"],
        "lang":        lang,
        "n_sentences": s["n_sentences"],
        "n_words":     s["n_words"],
        "asl":         float(s["asl"]),
        "asw":         float(s["asw"]),
        "flesch":      float(s["flesch"]),
        "fk_grade":    float(s["fk_grade"]),
        "kandel":      float(s["kandel"]),
    } for s in all_stats]))
    summary_df.write.mode("overwrite")               .parquet(os.path.join(parquet_dir, f"book_summaries{suffix}"))
    print(f"  Saved: book_summaries{suffix}")

    
    for s in all_stats:
        title_clean = s["title"].replace(" ", "_")
        wf          = s["word_freq"].copy()
        wf["title"] = s["title"]
        wf["genre"] = s["genre"]
        spark.createDataFrame(wf)              .write.mode("overwrite")              .parquet(os.path.join(parquet_dir, f"word_freq_{title_clean}{suffix}"))
    print(f"  Saved: individual word frequencies{suffix}")

    
    all_wf = pd.concat([
        s["word_freq"].assign(title=s["title"], genre=s["genre"])
        for s in all_stats
    ], ignore_index=True)
    spark.createDataFrame(all_wf)          .write.mode("overwrite")          .parquet(os.path.join(parquet_dir, f"all_word_frequencies{suffix}"))
    print(f"  Saved: all_word_frequencies{suffix}")

    
    all_segs = pd.concat([
        s["segments"].assign(title=s["title"], genre=s["genre"])
        for s in all_stats
    ], ignore_index=True)
    spark.createDataFrame(all_segs)          .write.mode("overwrite")          .parquet(os.path.join(parquet_dir, f"all_segments{suffix}"))
    print(f"  Saved: all_segments{suffix}")

    # 5 — Sentences
    all_sents = pd.concat([
        s["sentences"].assign(title=s["title"], genre=s["genre"])
        for s in all_stats
    ], ignore_index=True)
    spark.createDataFrame(all_sents)          .write.mode("overwrite")          .parquet(os.path.join(parquet_dir, f"all_sentences{suffix}"))
    print(f"  Saved: all_sentences{suffix}")

    
    corpus_df = spark.createDataFrame(pd.DataFrame([{
        "genre":       c["genre"],
        "lang":        lang,
        "n_sentences": c["n_sentences"],
        "n_words":     c["n_words"],
        "asl":         float(c["asl"]),
        "asw":         float(c["asw"]),
        "flesch":      float(c["flesch"]),
        "fk_grade":    float(c["fk_grade"]),
        "kandel":      float(c["kandel"]),
        "ttr":         float(c["ttr"]),
    } for c in [romantic_corpus, realist_corpus]]))
    corpus_df.write.mode("overwrite")              .parquet(os.path.join(parquet_dir, f"corpus_summaries{suffix}"))
    print(f"  Saved: corpus_summaries{suffix}")

    print(f"\nAll parquet files saved with suffix '{suffix}'!")
    


# In[23]:


def weighted_average(stats, metric, weight="n_words"):
    total_weight = sum(s[weight] for s in stats)
    return sum(s[metric] * s[weight] for s in stats) / total_weight

def plot_scatter_comparison(all_stats, save_dir=None,display=False,lang="en"):
    df = pd.DataFrame([{
        "title":  s["title"],
        "genre":  s["genre"],
        "flesch": s["flesch"],
        "fk":     s["fk_grade"],
        "asl":    s["asl"],
        "n_words":s["n_words"],
    } for s in all_stats])

    color_map = {"Romantique": "#e74c3c", "Réaliste": "#2980b9"}

    # Flesch vs FK Grade, bubble size = word count
    fig = go.Figure()
    for genre, color in color_map.items():
        subset = df[df["genre"] == genre]
        fig.add_trace(go.Scatter(
            x=subset["flesch"],
            y=subset["fk"],
            mode="markers+text",
            name=genre,
            text=subset["title"],
            textposition="top center",
            marker=dict(
                color=color,
                size=subset["n_words"] / 5000,  # scale bubble to word count
                sizemode="area",
                opacity=0.7,
                line=dict(width=1, color="white")
            )
        ))

    fig.update_layout(
        title="Flesch Ease vs FK Grade — Romantic vs Realist<br><sup>Bubble size = word count</sup>",
        xaxis_title="Flesch Reading Ease (higher = easier)",
        yaxis_title="FK Grade Level (higher = harder)",
        template="plotly_white",
        height=600
    )
    
    if save_dir :
        fig.write_html(os.path.join(save_dir, f"scatter_comparison_{lang}.html"))
        print("Saved: scatter_comparison.html")
        
    if display:
        fig.show()


# In[21]:


romantiques_en = [(696,"The Castle of Otranto"),(5998,"Waverley"), (82,"Ivanhoe"),(6406,"The Monastery"),(42389,"The Pirate"),(161,"Sense and Sensibility"),(1342,"Pride and Prejudice"),(158,"Emma"),(768,"Wuthering Heights"),(1260,"Jane Eyre"),(1028,"The Professor"),(2701,"Moby Dick"),(345,"Dracula")]

realistes_en = [(76,"The Adventures of Huckleberry Finn"),(74,"The Adventures of Tom Sawyer"),(145,"Middlemarch"), (1837,"The Prince and The Pauper"),(730,"Oliver Twist"), (1023,"Bleak House"),(766,"David Copperfield"),(98,"A Tale of Two Cities"),(1400,"Great Expectations"), ([2833,2834],"The Portrait of a Lady"),]

#le corpus réaliste ressemble un peu à un corpus Charles Dickens...


romantiques_fr = [
    # Chateaubriand
    (799,   "Atala"),
    (2473,  "René"),
    # Victor Hugo
    (135,   "Les Misérables"),
    (14287, "Notre-Dame de Paris"),
    # Alexandre Dumas
    (1257,  "Les Trois Mousquetaires"),
    (1258,  "Le Comte de Monte-Cristo"),
    # Stendhal
    (44747, "Le Rouge et le Noir"),
    (16942, "La Chartreuse de Parme"),
    # George Sand
    (5672,  "La Mare au Diable"),
    (3533,  "Indiana"),
    # Eugène Sue
    (4618,  "Les Mystères de Paris"),
    # Balzac
    (1237,  "Le Père Goriot"),
    (1184,  "Eugénie Grandet"),
    (20000, "La Cousine Bette"),
    (1320,  "Le Colonel Chabert"),
]

realistes_fr = [
    # Flaubert
    (4650,  "Madame Bovary"),
    (6581,  "L'Éducation Sentimentale"),
    (6509,  "Salammbô"),
    # Maupassant
    (547,   "Bel Ami"),
    (3024,  "Une Vie"),
    (23997, "Pierre et Jean"),
    # Zola
    (17450, "Germinal"),
    (5711,  "Nana"),
    (1069,  "L'Assommoir"),
    (3268,  "Au Bonheur des Dames"),
    # Huysmans
    (12341, "À Rebours"),
    # Alphonse Daudet
    (926,   "Tartarin de Tarascon"),
]


# In[22]:


stats_romantiques_en = analyze_corpus_books(romantiques_en,"Romantique")
stats_realistes_en = analyze_corpus_books(realistes_en,"Réaliste")
stats_en = stats_romantiques_en + stats_realistes_en

corpus_romantique_en = analyze_full_corpus_from_stats(stats_romantiques_en,"Romantique")
corpus_realiste_en = analyze_full_corpus_from_stats(stats_realistes_en,"Réaliste")

save_results_parquet(stats_en,corpus_romantique_en,corpus_realiste_en, lang="en")


# In[ ]:


stats_romantiques_fr  = analyze_corpus_books_fr(romantiques_fr, "Romantique_FR")
stats_realistes_fr    = analyze_corpus_books_fr(realistes_fr,   "Réaliste_FR")
stats_fr              = stats_romantiques_fr + stats_realistes_fr

corpus_romantique_fr  = analyze_full_corpus_from_stats(stats_romantiques_fr, "Romantique_FR")
corpus_realiste_fr    = analyze_full_corpus_from_stats(stats_realistes_fr,   "Réaliste_FR")

save_results_parquet(stats_fr, corpus_romantique_fr, corpus_realiste_fr, lang="fr")


# In[53]:


romantiques_fr = [
    (799,  "Atala"),                          # Chateaubriand
    (2473, "René"),                           # Chateaubriand
    (14287,"Notre-Dame de Paris"),            # Hugo
]

realistes_fr = [
    (4650, "Madame Bovary"),                  # Flaubert
    (547,  "Bel Ami"),                        # Maupassant
    (17450,"Germinal"),                       # Zola
]

romantic_stats_fr = analyze_corpus_books_fr(romantiques_fr, "Romantique")
realist_stats_fr  = analyze_corpus_books_fr(realistes_fr,   "Réaliste")

romantic_corpus_fr = analyze_full_corpus_from_stats(romantic_stats_fr, "Romantique")
realist_corpus_fr  = analyze_full_corpus_from_stats(realist_stats_fr,  "Réaliste")

# Compare
plot_books_comparison(romantic_stats_fr + realist_stats_fr,lang="fr")
plot_corpus_comparison(romantic_corpus_fr, realist_corpus_fr,lang="fr")



# In[17]:



    
    
    
def load_and_replot(lang="en", save_dir=os.path.expanduser(dir2)):
    parquet_dir = os.path.join(save_dir, "parquet_results")
    suffix      = f"_{lang}"

    book_summaries = spark.read.parquet(
        os.path.join(parquet_dir, f"book_summaries{suffix}")
    ).toPandas()

    all_word_freq = spark.read.parquet(
        os.path.join(parquet_dir, f"all_word_frequencies{suffix}")
    ).toPandas()

    all_segments = spark.read.parquet(
        os.path.join(parquet_dir, f"all_segments{suffix}")
    ).toPandas()

    all_sentences = spark.read.parquet(
        os.path.join(parquet_dir, f"all_sentences{suffix}")
    ).toPandas()

    # Pick correct syllable function based on lang
    syllable_fn = count_syllables_fr if lang == "fr" else count_syllables

    all_stats = []
    for _, row in book_summaries.iterrows():
        title = row["title"]
        wf    = all_word_freq[all_word_freq["title"] == title].copy()
        segs  = all_segments[all_segments["title"]  == title].copy()
        sents = all_sentences[all_sentences["title"] == title].copy()

        sentences_list = [
            s for s in sents["sentence"].tolist()
            if len(s.split()) < 500
        ]
        window_results = compute_readability_windows(
            sentences_list, [10, 25, 50, 100],
            syllable_fn=syllable_fn
        )

        all_stats.append({
            "title":       title,
            "genre":       row["genre"],
            "n_sentences": row["n_sentences"],
            "n_words":     row["n_words"],
            "asl":         row["asl"],
            "asw":         row["asw"],
            "flesch":      row["flesch"],
            "fk_grade":    row["fk_grade"],
            "kandel":      row["kandel"],
            "word_freq":   wf,
            "segments":    segs,
            "sentences":   sents
        })

        dialog_stats    = readability_for_group(segs[segs["type"] == "Dialog"])
        narration_stats = readability_for_group(segs[segs["type"] == "Narration"])

        _plot_readability(window_results, title, save_dir)
        _plot_zipf(wf, title, save_dir)
        _plot_segments(segs, dialog_stats, narration_stats, title, save_dir)
        _plot_word_freq(wf, title, save_dir)

    romantic_stats = [s for s in all_stats if "Romantique" in s["genre"]]
    realist_stats  = [s for s in all_stats if "Réaliste"   in s["genre"]]

    romantic_corpus = analyze_full_corpus_from_stats(romantic_stats, "Romantique")
    realist_corpus  = analyze_full_corpus_from_stats(realist_stats,  "Réaliste")

    plot_books_comparison(all_stats, save_dir)
    plot_corpus_comparison(romantic_corpus, realist_corpus, save_dir)
    plot_scatter_comparison(all_stats, save_dir)

    print(f"All '{lang}' plots regenerated!")
    return all_stats

def load_and_display(lang="en", save_dir=os.path.expanduser(dir2)):
    import plotly.io as pio
    pio.renderers.default = "notebook"

    parquet_dir = os.path.join(save_dir, "parquet_results")
    suffix      = f"_{lang}"

    # Load all data
    print(f"Loading {lang} data from parquet...")
    book_summaries = spark.read.parquet(
        os.path.join(parquet_dir, f"book_summaries{suffix}")
    ).toPandas()

    all_word_freq = spark.read.parquet(
        os.path.join(parquet_dir, f"all_word_frequencies{suffix}")
    ).toPandas()

    all_segments = spark.read.parquet(
        os.path.join(parquet_dir, f"all_segments{suffix}")
    ).toPandas()

    all_sentences = spark.read.parquet(
        os.path.join(parquet_dir, f"all_sentences{suffix}")
    ).toPandas()

    syllable_fn = count_syllables_fr if lang == "fr" else count_syllables
    stopwords_l = stopwords_fr       if lang == "fr" else stopwords

    # ----------------------------------------------------------------
    # Reconstruct all_stats
    # ----------------------------------------------------------------
    all_stats = []
    for _, row in book_summaries.iterrows():
        title = row["title"]
        wf    = all_word_freq[all_word_freq["title"] == title].copy()
        segs  = all_segments[all_segments["title"]   == title].copy()
        sents = all_sentences[all_sentences["title"] == title].copy()

        sentences_list = [
            s for s in sents["sentence"].tolist()
            if len(s.split()) < 500
        ]
        window_results = compute_readability_windows(
            sentences_list, [10, 25, 50, 100],
            syllable_fn=syllable_fn
        )

        all_stats.append({
            "title":       title,
            "genre":       row["genre"],
            "n_sentences": row["n_sentences"],
            "n_words":     row["n_words"],
            "asl":         row["asl"],
            "asw":         row["asw"],
            "flesch":      row["flesch"],
            "fk_grade":    row["fk_grade"],
            "kandel":      row["kandel"],
            "word_freq":   wf,
            "segments":    segs,
            "sentences":   sents,
            "windows":     window_results
        })

    # ----------------------------------------------------------------
    # Per-book plots — displayed inline
    # ----------------------------------------------------------------
    for s in all_stats:
        title          = s["title"]
        wf             = s["word_freq"]
        segs           = s["segments"]
        window_results = s["windows"]

        print(f"\n{'='*60}")
        print(f"  {title}  ({s['genre']})")
        print(f"  Sentences: {s['n_sentences']}  |  Words: {s['n_words']}")
        print(f"  Flesch: {s['flesch']:.2f}  |  FK: {s['fk_grade']:.2f}  |  Kandel: {s['kandel']:.2f}")
        print(f"{'='*60}")

        # --- Readability sliding windows ---
        fig = make_subplots(rows=3, cols=1,
            subplot_titles=("Flesch Reading Ease",
                            "Flesch-Kincaid Grade Level",
                            "Kandel-Moles"),
            shared_xaxes=True)
        colors = {10: "#e74c3c", 25: "#e67e22", 50: "#2980b9", 100: "#27ae60"}
        for ws, df_w in window_results.items():
            for row, metric in enumerate(["flesch", "fk_grade", "kandel_moles"], start=1):
                fig.add_trace(go.Scatter(
                    x=df_w["window_mid"], y=df_w[metric],
                    mode="lines", name=f"Window={ws}",
                    line=dict(color=colors[ws], width=1.5),
                    legendgroup=f"ws{ws}",
                    showlegend=(row == 1)
                ), row=row, col=1)
        fig.add_hline(y=60, line_dash="dash", line_color="gray",
                      annotation_text="Standard (60)", row=1, col=1)
        fig.update_layout(height=700,
                          title_text=f"Readability Stability — {title}",
                          template="plotly_white")
        fig.show()

        # --- Word frequency ---
        top30 = wf[~wf["word"].isin(stopwords_l)]         .sort_values("count", ascending=False)         .head(30)
        fig2  = px.bar(top30, x="word", y="count",
                       title=f"Top 30 Words — {title}",
                       template="plotly_white",
                       color_discrete_sequence=["#e07b39"])
        fig2.show()

        # --- Zipf ---
        wf_zipf = wf.copy()
        wf_zipf["rank"]       = range(1, len(wf_zipf) + 1)
        wf_zipf["zipf_ideal"] = wf_zipf["count"].max() / wf_zipf["rank"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=wf_zipf["rank"], y=wf_zipf["count"],
            mode="lines", name="Observed",
            line=dict(color="#e74c3c", width=2)))
        fig3.add_trace(go.Scatter(
            x=wf_zipf["rank"], y=wf_zipf["zipf_ideal"],
            mode="lines", name="Ideal Zipf",
            line=dict(color="#2980b9", dash="dash")))
        fig3.update_layout(
            title=f"Zipf Plot — {title}",
            xaxis=dict(title="Rank", type="log"),
            yaxis=dict(title="Frequency", type="log"),
            template="plotly_white")
        fig3.show()

        # --- Dialog vs Narration ---
        dialog_stats    = readability_for_group(segs[segs["type"] == "Dialog"])
        narration_stats = readability_for_group(segs[segs["type"] == "Narration"])

        fig4 = make_subplots(rows=1, cols=3,
            subplot_titles=("Sentence Count", "Avg Length", "Flesch Ease"),
            specs=[[{"type":"xy"}]*3])
        for col_idx, metric in enumerate(["count", "avg_length", "flesch"], start=1):
            fig4.add_trace(go.Bar(
                x=["Dialog", "Narration"],
                y=[dialog_stats.get(metric, 0), narration_stats.get(metric, 0)],
                marker_color=["#e74c3c", "#2980b9"],
                showlegend=False
            ), row=1, col=col_idx)
        fig4.update_layout(title_text=f"Dialog vs Narration — {title}",
                           template="plotly_white")
        fig4.show()

        # --- Rolling dialog ratio ---
        segs["dialog_ratio"] = (
            (segs["type"] == "Dialog").astype(int).rolling(50).mean()
        )
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=segs["idx"], y=segs["dialog_ratio"],
            mode="lines", fill="tozeroy",
            line=dict(color="#e74c3c"), name="Dialog Ratio"))
        fig5.update_layout(
            title=f"Dialog Proportion — {title}",
            xaxis_title="Sentence Index",
            yaxis_title="Proportion",
            yaxis=dict(range=[0, 1]),
            template="plotly_white")
        fig5.show()

    # ----------------------------------------------------------------
    # Corpus-level plots
    # ----------------------------------------------------------------
    romantic_stats  = [s for s in all_stats if "Romantique" in s["genre"]]
    realist_stats   = [s for s in all_stats if "Réaliste"   in s["genre"]]
    romantic_corpus = analyze_full_corpus_from_stats(romantic_stats, "Romantique")
    realist_corpus  = analyze_full_corpus_from_stats(realist_stats,  "Réaliste")

    print("\n" + "="*60)
    print("  CORPUS-LEVEL COMPARISON")
    print("="*60)

    # Books comparison
    plot_books_comparison(all_stats)

    # Corpus comparison + Zipf
    plot_corpus_comparison(romantic_corpus, realist_corpus)

    # Scatter
    plot_scatter_comparison(all_stats)

    print("\nDone!")
    return all_stats


# In[ ]:





# In[27]:


ulysses = load_gutenberg_book(4300,"Ulysses")
ulysses_stats = analyze_book(ulysses,"Ulysses",display=True)


# In[19]:


mobydick = load_gutenberg_book(2701,"Moby Dick")
mobydick_stats = analyze_book(mobydick, "Moby Dick")


# In[15]:





# In[20]:


prideandprejudice = load_gutenberg_book(1342, "Pride and Prejudice")
prideandprejudice_stats = analyze_book(prideandprejudice,"Pride and Prejudice")


# In[54]:





# In[67]:


romantic_stats = analyze_corpus_books(romantiques_en, "Romantique")
realist_stats  = analyze_corpus_books(realistes_en,   "Réaliste")
all_stats      = romantic_stats + realist_stats


# In[68]:



plot_books_comparison(all_stats)


# In[69]:


romantic_corpus = analyze_full_corpus_from_stats(romantic_stats,"Romantique")
realist_corpus  = analyze_full_corpus_from_stats(realist_stats,   "Réaliste")  
save_results_parquet(all_stats, romantic_corpus, realist_corpus)


# In[51]:





# In[72]:


all_stats_en = load_and_replot(lang="en")


# In[34]:


all_stats_en2 = load_and_display(lang="en")


# In[39]:


def quarto_book_explorer(all_stats, lang="en"):
    """
    Single figure with a dropdown to switch between books.
    Works in static Quarto HTML — no kernel needed.
    """
    stopwords_l = stopwords_fr if lang == "fr" else stopwords

    # Build one figure with all books as traces, toggle via dropdown
    fig  = go.Figure()
    buttons = []

    for i, s in enumerate(all_stats):
        wf    = s["word_freq"][~s["word_freq"]["word"].isin(stopwords_l)]                 .sort_values("count", ascending=False).head(30)
        visible = (i == 0)

        fig.add_trace(go.Bar(
            x=wf["word"],
            y=wf["count"],
            name=s["title"],
            visible=visible,
            marker_color="#e07b39"
        ))

        buttons.append(dict(
            label=f"{s['title']} ({s['genre']})",
            method="update",
            args=[
                {"visible": [j == i for j in range(len(all_stats))]},
                {"title": f"Top 30 Words — {s['title']}"}
            ]
        ))

    fig.update_layout(
        title=f"Top 30 Words — {all_stats[0]['title']}",
        template="plotly_white",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            y=1.15
        )]
    )
    fig.show()


def quarto_zipf_explorer(all_stats):
    """Zipf plots for all books with dropdown — works in Quarto HTML."""
    fig     = go.Figure()
    buttons = []

    for i, s in enumerate(all_stats):
        wf               = s["word_freq"].copy()
        wf["rank"]       = range(1, len(wf) + 1)
        wf["zipf_ideal"] = wf["count"].max() / wf["rank"]
        visible          = (i == 0)

        fig.add_trace(go.Scatter(
            x=wf["rank"], y=wf["count"],
            mode="lines", name=f"{s['title']} (observed)",
            visible=visible,
            line=dict(color="#e74c3c", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=wf["rank"], y=wf["zipf_ideal"],
            mode="lines", name=f"{s['title']} (ideal Zipf)",
            visible=visible,
            line=dict(color="#2980b9", dash="dash")
        ))

        # Each book has 2 traces
        visibility = [False] * (len(all_stats) * 2)
        visibility[i*2]   = True
        visibility[i*2+1] = True

        buttons.append(dict(
            label=f"{s['title']} ({s['genre']})",
            method="update",
            args=[
                {"visible": visibility},
                {"title": f"Zipf Plot — {s['title']}"}
            ]
        ))

    fig.update_layout(
        title=f"Zipf Plot — {all_stats[0]['title']}",
        xaxis=dict(title="Rank", type="log"),
        yaxis=dict(title="Frequency", type="log"),
        template="plotly_white",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            showactive=True,
            x=0.0,
            y=1.15
        )]
    )
    fig.show()
    
    
    
def quarto_book_explorer_full(all_stats, lang="en"):
    """
    Full per-book analysis displayed with Plotly dropdown menus.
    Works in static Quarto HTML — no kernel needed.
    """
    stopwords_l = stopwords_fr if lang == "fr" else stopwords

    # ----------------------------------------------------------------
    # 1 — Word Frequency
    # ----------------------------------------------------------------
    fig_wf   = go.Figure()
    fig_zipf = go.Figure()
    fig_read = make_subplots(rows=3, cols=1,
        subplot_titles=("Flesch Reading Ease",
                        "Flesch-Kincaid Grade Level",
                        "Kandel-Moles"),
        shared_xaxes=False)
    fig_dial  = make_subplots(rows=1, cols=3,
        subplot_titles=("Sentence Count", "Avg Length", "Flesch Ease"),
        specs=[[{"type":"xy"}]*3])
    fig_ratio = go.Figure()

    buttons_wf   = []
    buttons_zipf = []
    buttons_read = []
    buttons_dial = []
    buttons_ratio= []

    # Track trace counts per figure for visibility toggling
    wf_traces_per_book   = 1
    zipf_traces_per_book = 2
    read_traces_per_book = None  # computed per book (4 window sizes x 3 subplots)
    dial_traces_per_book = 3     # 3 metrics
    ratio_traces_per_book= 1

    read_trace_counts = []

    colors = {10: "#e74c3c", 25: "#e67e22", 50: "#2980b9", 100: "#27ae60"}

    for i, s in enumerate(all_stats):
        visible = (i == 0)
        title   = s["title"]
        genre   = s["genre"]
        wf      = s["word_freq"]
        segs    = s["segments"].copy()
        windows = s["windows"]

        label = f"{title} ({genre})"

        # --- Word Frequency ---
        top30 = wf[~wf["word"].isin(stopwords_l)]             .sort_values("count", ascending=False).head(30)
        fig_wf.add_trace(go.Bar(
            x=top30["word"], y=top30["count"],
            name=label, visible=visible,
            marker_color="#e07b39",
            showlegend=False
        ))

        # --- Zipf ---
        wf_z               = wf.copy()
        wf_z["rank"]       = range(1, len(wf_z) + 1)
        wf_z["zipf_ideal"] = wf_z["count"].max() / wf_z["rank"]
        fig_zipf.add_trace(go.Scatter(
            x=wf_z["rank"], y=wf_z["count"],
            mode="lines", name="Observed",
            visible=visible,
            line=dict(color="#e74c3c", width=2),
            showlegend=visible
        ))
        fig_zipf.add_trace(go.Scatter(
            x=wf_z["rank"], y=wf_z["zipf_ideal"],
            mode="lines", name="Ideal Zipf",
            visible=visible,
            line=dict(color="#2980b9", dash="dash"),
            showlegend=visible
        ))

        # --- Readability sliding windows ---
        n_read_traces = 0
        for ws, df_w in windows.items():
            for row_idx, metric in enumerate(
                    ["flesch", "fk_grade", "kandel_moles"], start=1):
                fig_read.add_trace(go.Scatter(
                    x=df_w["window_mid"], y=df_w[metric],
                    mode="lines",
                    name=f"Window={ws}",
                    visible=visible,
                    line=dict(color=colors[ws], width=1.5),
                    legendgroup=f"ws{ws}_{i}",
                    showlegend=(row_idx == 1 and visible)
                ), row=row_idx, col=1)
                n_read_traces += 1
        read_trace_counts.append(n_read_traces)

        # --- Dialog vs Narration ---
        dialog_stats    = readability_for_group(segs[segs["type"] == "Dialog"])
        narration_stats = readability_for_group(segs[segs["type"] == "Narration"])
        for col_idx, metric in enumerate(
                ["count", "avg_length", "flesch"], start=1):
            fig_dial.add_trace(go.Bar(
                x=["Dialog", "Narration"],
                y=[dialog_stats.get(metric, 0),
                   narration_stats.get(metric, 0)],
                marker_color=["#e74c3c", "#2980b9"],
                name=label,
                visible=visible,
                showlegend=False
            ), row=1, col=col_idx)

        # --- Dialog Ratio ---
        segs["dialog_ratio"] = (
            (segs["type"] == "Dialog")
            .astype(int).rolling(50).mean()
        )
        fig_ratio.add_trace(go.Scatter(
            x=segs["idx"], y=segs["dialog_ratio"],
            mode="lines", fill="tozeroy",
            name=label,
            visible=visible,
            line=dict(color="#e74c3c"),
            showlegend=False
        ))

    # ----------------------------------------------------------------
    # Build dropdown buttons
    # ----------------------------------------------------------------
    n_books = len(all_stats)

    for i, s in enumerate(all_stats):
        label = f"{s['title']} ({s['genre']})"

        # Word frequency — 1 trace per book
        vis_wf = [j == i for j in range(n_books)]
        buttons_wf.append(dict(
            label=label, method="update",
            args=[{"visible": vis_wf},
                  {"title": f"Top 30 Words — {s['title']}"}]
        ))

        # Zipf — 2 traces per book
        vis_zipf = [False] * (n_books * 2)
        vis_zipf[i*2]   = True
        vis_zipf[i*2+1] = True
        buttons_zipf.append(dict(
            label=label, method="update",
            args=[{"visible": vis_zipf},
                  {"title": f"Zipf Plot — {s['title']}"}]
        ))

        # Readability — variable traces per book
        vis_read  = []
        offset    = 0
        for j, count in enumerate(read_trace_counts):
            for _ in range(count):
                vis_read.append(j == i)
            offset += count
        buttons_read.append(dict(
            label=label, method="update",
            args=[{"visible": vis_read},
                  {"title": f"Readability — {s['title']}"}]
        ))

        # Dialog bars — 3 traces per book
        vis_dial = [False] * (n_books * 3)
        for k in range(3):
            vis_dial[i*3 + k] = True
        buttons_dial.append(dict(
            label=label, method="update",
            args=[{"visible": vis_dial},
                  {"title": f"Dialog vs Narration — {s['title']}"}]
        ))

        # Dialog ratio — 1 trace per book
        vis_ratio = [j == i for j in range(n_books)]
        buttons_ratio.append(dict(
            label=label, method="update",
            args=[{"visible": vis_ratio},
                  {"title": f"Dialog Proportion — {s['title']}"}]
        ))

    # ----------------------------------------------------------------
    # Apply dropdowns and layouts
    # ----------------------------------------------------------------
    dropdown = dict(direction="down", showactive=True, x=0.0, y=1.15)

    fig_wf.update_layout(
        title=f"Top 30 Words — {all_stats[0]['title']}",
        template="plotly_white",
        updatemenus=[{**dropdown, "buttons": buttons_wf, "active": 0}]
    )

    fig_zipf.update_layout(
        title=f"Zipf Plot — {all_stats[0]['title']}",
        xaxis=dict(title="Rank", type="log"),
        yaxis=dict(title="Frequency", type="log"),
        template="plotly_white",
        updatemenus=[{**dropdown, "buttons": buttons_zipf, "active": 0}]
    )

    fig_read.update_layout(
        height=700,
        title_text=f"Readability Stability — {all_stats[0]['title']}",
        template="plotly_white",
        updatemenus=[{**dropdown, "buttons": buttons_read, "active": 0}]
    )
    fig_read.add_hline(y=60, line_dash="dash", line_color="gray",
                       annotation_text="Standard (60)", row=1, col=1)
    fig_read.add_hline(y=30, line_dash="dot",  line_color="red",
                       annotation_text="Very Hard (30)", row=1, col=1)

    fig_dial.update_layout(
        title_text=f"Dialog vs Narration — {all_stats[0]['title']}",
        template="plotly_white",
        updatemenus=[{**dropdown, "buttons": buttons_dial, "active": 0}]
    )

    fig_ratio.update_layout(
        title=f"Dialog Proportion — {all_stats[0]['title']}",
        xaxis_title="Sentence Index",
        yaxis_title="Proportion of Dialog",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        updatemenus=[{**dropdown, "buttons": buttons_ratio, "active": 0}]
    )

    # ----------------------------------------------------------------
    # Display all
    # ----------------------------------------------------------------
    fig_read.show()
    fig_wf.show()
    fig_zipf.show()
    fig_dial.show()
    fig_ratio.show()


# In[40]:


quarto_book_explorer_full(all_stats_en2)


# In[45]:


romantic_stats  = [s for s in all_stats_en2 if "Romantique" in s["genre"]]
realist_stats   = [s for s in all_stats_en2 if "Réaliste"   in s["genre"]]


# In[53]:


romantic_corpus = analyze_full_corpus_from_stats(romantic_stats, "Romantique")
realist_corpus  = analyze_full_corpus_from_stats(realist_stats,  "Réaliste")
plot_books_comparison(all_stats_en2,
                              save_dir=None, display=True)
plot_corpus_comparison(romantic_corpus, realist_corpus,
                               save_dir=None, display=True)
plot_scatter_comparison(all_stats_en2,
                                save_dir=None, display=True)


# In[42]:


# Load book summaries
book_summaries = spark.read.parquet(
    os.path.join(os.path.expanduser(dir2), "parquet_results", "book_summaries_en")
)
book_summaries.show()

# Load corpus summaries
corpus_summaries = spark.read.parquet(
    os.path.join(os.path.expanduser(dir2), "parquet_results", "corpus_summaries_en")
)
corpus_summaries.show()


# In[63]:


plot_corpus_comparison(romantic_corpus,realist_corpus)


# In[11]:


# Requires a pretrained POS model
pos_tagger = PerceptronModel.pretrained("pos_anc", "en")     .setInputCols(["sentence", "token"])     .setOutputCol("pos")

pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos_tagger])
result = pipeline.fit(df).transform(df)



# In[12]:


# Explode token array into individual rows
# Step 1 — Collapse all lines into a single document
full_text = df.select(
    concat_ws(" ", collect_list("text")).alias("text")
)

# Step 2 — Run pipeline on the full text
result = pipeline.fit(full_text).transform(full_text)

# Step 3 — Extract tokens and POS tags
from pyspark.sql.functions import posexplode

tokens_df = result.select(
    posexplode(col("token")).alias("idx", "tok")
).select(
    col("idx"),
    col("tok.result").alias("word")
)

pos_df = result.select(
    posexplode(col("pos")).alias("idx", "p")
).select(
    col("idx"),
    col("p.result").alias("POS_tag")
)

tokens_df.join(pos_df, on="idx").show(20, truncate=False)


# In[13]:


tokens_df.join(pos_df, on="idx").show(200, truncate=False)


# In[14]:


# Embeddings are required before NER
embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")     .setInputCols(["sentence", "token"])     .setOutputCol("embeddings")

ner = NerDLModel.pretrained("ner_dl", "en")     .setInputCols(["sentence", "token", "embeddings"])     .setOutputCol("ner")

# Convert NER output to readable spans
ner_converter = NerConverter()     .setInputCols(["sentence", "token", "ner"])     .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, sentence_detector, tokenizer,
    embeddings, ner, ner_converter
])

result = pipeline.fit(df).transform(df)

# Show named entities (persons, locations, organizations...)
result.select(
    explode("ner_chunk").alias("chunk")
).select(
    col("chunk.result").alias("entity"),
    col("chunk.metadata.entity").alias("label")
).show(20, truncate=False)


# In[15]:


from pyspark.sql.functions import col, explode

lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en")     .setInputCols(["token"])     .setOutputCol("lemma")

stopwords_cleaner = StopWordsCleaner.pretrained("stopwords_en", "en")     .setInputCols(["lemma"])     .setOutputCol("clean_tokens")     .setCaseSensitive(False)

pipeline = Pipeline(stages=[
    document_assembler, sentence_detector, tokenizer,
    lemmatizer, stopwords_cleaner
])

result = pipeline.fit(df).transform(df)

# Show clean lemmatized tokens
result.select(explode("clean_tokens.result").alias("clean_word"))       .filter(col("clean_word") != "")       .show(20)


# In[16]:


pipeline = Pipeline(stages=[
    DocumentAssembler().setInputCol("text").setOutputCol("document"),
    SentenceDetector().setInputCols(["document"]).setOutputCol("sentence"),
    Tokenizer().setInputCols(["sentence"]).setOutputCol("token"),
    LemmatizerModel.pretrained("lemma_antbnc", "en")
        .setInputCols(["token"]).setOutputCol("lemma"),
    StopWordsCleaner.pretrained("stopwords_en", "en")
        .setInputCols(["lemma"]).setOutputCol("clean_tokens"),
    PerceptronModel.pretrained("pos_anc", "en")
        .setInputCols(["sentence", "token"]).setOutputCol("pos"),
    WordEmbeddingsModel.pretrained("glove_100d", "en")
        .setInputCols(["sentence", "token"]).setOutputCol("embeddings"),
    NerDLModel.pretrained("ner_dl", "en")
        .setInputCols(["sentence", "token", "embeddings"]).setOutputCol("ner"),
    NerConverter()
        .setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")
])

result = pipeline.fit(df).transform(df)
result.printSchema()


# In[17]:


from pyspark.sql.functions import (
    col, explode, count, avg, length, 
    collect_list, size, udf, countDistinct
)
from pyspark.sql.types import FloatType
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# In[23]:


stopwords = [
    "the", "a", "an", "and", "of", "to", "in", "was", "is", "are",
    "he", "she", "it", "that", "her", "his", "had", "i", "be", "as",
    "at", "by", "we", "or", "but", "not", "with", "for", "on", "so",
    "my", "you", "have", "from", "they", "all", "me", "no", "do", "if",
    "been", "would", "could", "should", "than", "then", "them", "their",
    "what", "which", "who", "this", "were", "has", "one", "when", "up"
]






# --- 1. Average Sentence Length (in tokens) ---
sentence_lengths = result.select(
    explode(col("sentence")).alias("sent")
).select(
    length(col("sent.result")).alias("char_length")
)

avg_sent_length = sentence_lengths.agg(
    avg("char_length").alias("avg_sentence_length")
).collect()[0]["avg_sentence_length"]

# --- 2. Word Frequency ---
word_freq = result.select(
    explode(col("token.result")).alias("word")
).filter(
    col("word").rlike("^[a-zA-Z]+$") & # letters only
    ~col("word").isin(stopwords)  
).groupBy("word") \
 .count() \
 .orderBy(col("count").desc())

# --- 3. POS Distribution ---
pos_dist = result.select(
    explode(col("pos.result")).alias("pos_tag")
).groupBy("pos_tag") \
 .count() \
 .orderBy(col("count").desc())

# --- 4. Type-Token Ratio (Lexical Diversity) ---
total_tokens = result.select(
    explode(col("token.result")).alias("word")
).filter(col("word").rlike("^[a-zA-Z]+$")).count()

unique_tokens = result.select(
    explode(col("token.result")).alias("word")
).filter(col("word").rlike("^[a-zA-Z]+$")) \
 .select(col("word")).distinct().count()

ttr = unique_tokens / total_tokens

# --- 5. Word Length Distribution ---
word_lengths = result.select(
    explode(col("token.result")).alias("word")
).filter(col("word").rlike("^[a-zA-Z]+$")) \
 .select(length(col("word")).alias("word_length")) \
 .groupBy("word_length").count() \
 .orderBy("word_length")

# Convert to Pandas for Plotly
word_freq_pd    = word_freq.limit(30).toPandas()
pos_dist_pd     = pos_dist.limit(15).toPandas()
word_lengths_pd = word_lengths.toPandas()


# In[27]:


from pyspark.sql.functions import lower

stopwords = [
    "the", "a", "an", "and", "of", "to", "in", "was", "is", "are",
    "he", "she", "it", "that", "her", "his", "had", "i", "be", "as",
    "at", "by", "we", "or", "but", "not", "with", "for", "on", "so",
    "my", "you", "have", "from", "they", "all", "me", "no", "do", "if",
    "been", "would", "could", "should", "than", "then", "them", "their",
    "what", "which", "who", "this", "were", "has", "one", "when", "up",
    "said", "mr", "mrs", "miss", "s", "i", "t"  # ← add these
]

word_freq = result.select(
    explode(col("token.result")).alias("word")
).select(
    lower(col("word")).alias("word")          # ← lowercase first
).filter(
    col("word").rlike("^[a-zA-Z]{2,}$") &    # ← min 2 letters (removes "i", "_")
    ~col("word").isin(stopwords)
).groupBy("word").count() \
 .orderBy(col("count").desc())


word_freq_pd    = word_freq.limit(30).toPandas()


# In[28]:


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Top 30 Most Frequent Words",
        "POS Tag Distribution",
        "Word Length Distribution",
        "Lexical Diversity (TTR)"
    ),
    specs=[
        [{"type": "xy"},        {"type": "xy"}],
        [{"type": "xy"},        {"type": "indicator"}]  # ← add this
    ]
)

# --- Plot 1: Word Frequency ---
fig.add_trace(
    go.Bar(
        x=word_freq_pd["word"],
        y=word_freq_pd["count"],
        marker_color="#e07b39",
        name="Word Frequency"
    ),
    row=1, col=1
)

# --- Plot 2: POS Distribution ---
fig.add_trace(
    go.Bar(
        x=pos_dist_pd["pos_tag"],
        y=pos_dist_pd["count"],
        marker_color="#3a7ebf",
        name="POS Tags"
    ),
    row=1, col=2
)

# --- Plot 3: Word Length Distribution ---
fig.add_trace(
    go.Bar(
        x=word_lengths_pd["word_length"],
        y=word_lengths_pd["count"],
        marker_color="#2ca02c",
        name="Word Lengths"
    ),
    row=2, col=1
)

# --- Plot 4: TTR Gauge ---
fig.add_trace(
    go.Indicator(
        mode="gauge+number",
        value=round(ttr, 4),
        title={"text": "Type-Token Ratio"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#9467bd"},
            "steps": [
                {"range": [0, 0.3], "color": "#ffcccc"},
                {"range": [0.3, 0.6], "color": "#fff3cc"},
                {"range": [0.6, 1.0], "color": "#ccffcc"},
            ]
        }
    ),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="Stylometric Analysis — Pride and Prejudice",
    title_font_size=20,
    showlegend=False,
    template="plotly_white"
)


# In[29]:


fig.show(renderer="notebook")


# In[30]:


from pyspark.sql.functions import (
    explode, col, lower, length, collect_list, 
    concat_ws, avg, count, udf, pandas_udf
)
from pyspark.sql.types import FloatType, ArrayType, StringType, IntegerType
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# --- Count syllables in a word (approximation) ---
def count_syllables(word):
    word = word.lower().strip(".:;?!")
    if len(word) == 0:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

syllable_udf = udf(count_syllables, IntegerType())


# In[31]:


# Extract sentences and tokens
sentences_pd = result.select(
    explode(col("sentence")).alias("sent")
).select(col("sent.result").alias("sentence")).toPandas()

tokens_pd = result.select(
    explode(col("token.result")).alias("word")
).filter(col("word").rlike("^[a-zA-Z]+$")).toPandas()

# Compute syllables per word
tokens_pd["syllables"] = tokens_pd["word"].apply(count_syllables)

# Global stats
num_sentences   = len(sentences_pd)
num_words       = len(tokens_pd)
num_syllables   = tokens_pd["syllables"].sum()
avg_sent_length = num_words / num_sentences          # words per sentence
avg_syllables   = num_syllables / num_words          # syllables per word

# --- Flesch Reading Ease ---
# Higher = easier (60-70 is standard, <30 is very hard)
flesch_ease = 206.835 - (1.015 * avg_sent_length) - (84.6 * avg_syllables)

# --- Flesch-Kincaid Grade Level ---
# Corresponds to US school grade
fk_grade = (0.39 * avg_sent_length) + (11.8 * avg_syllables) - 15.59

# --- Kandel-Moles (French adaptation) ---
# Adapted for literary French/European texts
kandel_moles = 209.835 - (1.015 * avg_sent_length) - (84.6 * avg_syllables)

print(f"Sentences:             {num_sentences}")
print(f"Words:                 {num_words}")
print(f"Avg sentence length:   {avg_sent_length:.2f} words")
print(f"Avg syllables/word:    {avg_syllables:.2f}")
print(f"Flesch Reading Ease:   {flesch_ease:.2f}")
print(f"Flesch-Kincaid Grade:  {fk_grade:.2f}")
print(f"Kandel-Moles:          {kandel_moles:.2f}")


# In[32]:


def compute_readability_windows(sentences, window_sizes=[10, 25, 50, 100]):
    """Compute readability indices over sliding windows of sentences."""
    results = {}

    for ws in window_sizes:
        records = []
        for i in range(0, len(sentences) - ws, ws // 2):  # 50% overlap
            window = sentences[i:i + ws]
            words  = " ".join(window).split()
            n_sent = len(window)
            n_word = len(words)

            if n_word == 0 or n_sent == 0:
                continue

            n_syl       = sum(count_syllables(w) for w in words)
            asl         = n_word / n_sent
            asw         = n_syl  / n_word
            flesch      = 206.835 - (1.015 * asl) - (84.6  * asw)
            fk          = (0.39  * asl) + (11.8  * asw) - 15.59
            kandel      = 209.835 - (1.015 * asl) - (84.6  * asw)

            records.append({
                "window_start": i,
                "window_mid":   i + ws // 2,
                "flesch":       flesch,
                "fk_grade":     fk,
                "kandel_moles": kandel
            })

        results[ws] = pd.DataFrame(records)
    return results

sentences_list = sentences_pd["sentence"].tolist()
window_results = compute_readability_windows(sentences_list, [10, 25, 50, 100])


# In[33]:


fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        "Flesch Reading Ease — Sliding Windows",
        "Flesch-Kincaid Grade Level — Sliding Windows",
        "Kandel-Moles — Sliding Windows"
    ),
    shared_xaxes=True
)

colors = {10: "#e74c3c", 25: "#e67e22", 50: "#2980b9", 100: "#27ae60"}

for ws, df_w in window_results.items():
    # Flesch
    fig.add_trace(go.Scatter(
        x=df_w["window_mid"], y=df_w["flesch"],
        mode="lines", name=f"Window={ws}",
        line=dict(color=colors[ws], width=1.5),
        legendgroup=f"ws{ws}"
    ), row=1, col=1)

    # FK Grade
    fig.add_trace(go.Scatter(
        x=df_w["window_mid"], y=df_w["fk_grade"],
        mode="lines", name=f"Window={ws}",
        line=dict(color=colors[ws], width=1.5),
        legendgroup=f"ws{ws}", showlegend=False
    ), row=2, col=1)

    # Kandel-Moles
    fig.add_trace(go.Scatter(
        x=df_w["window_mid"], y=df_w["kandel_moles"],
        mode="lines", name=f"Window={ws}",
        line=dict(color=colors[ws], width=1.5),
        legendgroup=f"ws{ws}", showlegend=False
    ), row=3, col=1)

# Add reference lines for Flesch
fig.add_hline(y=60, line_dash="dash", line_color="gray",
              annotation_text="Standard (60)", row=1, col=1)
fig.add_hline(y=30, line_dash="dot",  line_color="red",
              annotation_text="Very Hard (30)", row=1, col=1)

fig.update_layout(
    height=900,
    title_text="Readability Stability — Pride and Prejudice",
    template="plotly_white"
)
fig.write_html(r"C:/Users/Utilisateur/Documents/readability_sliding.html")
print("Saved!")


# In[34]:


import re

def segment_text(sentences):
    """Classify each sentence as dialog or narration."""
    records = []
    for i, sent in enumerate(sentences):
        s = sent.strip()
        # Dialog: starts/ends with quotes, or contains speech verbs
        is_dialog = bool(re.match(r'^["\u201c\u2018]', s)) or                     bool(re.search(r'["\u201d\u2019]$', s)) or                     bool(re.search(r'\b(said|replied|asked|cried|exclaimed|answered|whispered|shouted)\b', s, re.I))
        records.append({
            "idx":      i,
            "sentence": s,
            "type":     "Dialog" if is_dialog else "Narration",
            "length":   len(s.split()),
            "syllables": sum(count_syllables(w) for w in s.split())
        })
    return pd.DataFrame(records)

segments_df = segment_text(sentences_list)

# Readability per segment type
def readability_for_group(group_df):
    n_sent = len(group_df)
    n_word = group_df["length"].sum()
    n_syl  = group_df["syllables"].sum()
    if n_word == 0 or n_sent == 0:
        return {}
    asl = n_word / n_sent
    asw = n_syl  / n_word
    return {
        "count":        n_sent,
        "avg_length":   asl,
        "flesch":       206.835 - (1.015 * asl) - (84.6 * asw),
        "fk_grade":     (0.39  * asl) + (11.8  * asw) - 15.59,
        "kandel_moles": 209.835 - (1.015 * asl) - (84.6 * asw)
    }

dialog_stats    = readability_for_group(segments_df[segments_df["type"] == "Dialog"])
narration_stats = readability_for_group(segments_df[segments_df["type"] == "Narration"])

print("Dialog:   ", dialog_stats)
print("Narration:", narration_stats)


# In[35]:


fig_seg = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Sentence Count", "Avg Sentence Length", "Flesch Reading Ease"),
    specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
)

categories = ["Dialog", "Narration"]
colors_seg = ["#e74c3c", "#2980b9"]

for col_idx, (metric, label) in enumerate([
    ("count",      "Sentence Count"),
    ("avg_length", "Avg Length (words)"),
    ("flesch",     "Flesch Ease")
], start=1):
    fig_seg.add_trace(go.Bar(
        x=categories,
        y=[dialog_stats[metric], narration_stats[metric]],
        marker_color=colors_seg,
        name=label,
        showlegend=False
    ), row=1, col=col_idx)

# Dialog proportion over the book (rolling)
segments_df["is_dialog"] = (segments_df["type"] == "Dialog").astype(int)
segments_df["dialog_ratio"] = segments_df["is_dialog"].rolling(50).mean()

fig_seg2 = go.Figure()
fig_seg2.add_trace(go.Scatter(
    x=segments_df["idx"],
    y=segments_df["dialog_ratio"],
    mode="lines",
    fill="tozeroy",
    name="Dialog Ratio",
    line=dict(color="#e74c3c")
))
fig_seg2.update_layout(
    title="Dialog Proportion Across the Novel (Rolling Window=50)",
    xaxis_title="Sentence Index",
    yaxis_title="Proportion of Dialog",
    template="plotly_white",
    yaxis=dict(range=[0, 1])
)

fig_seg.update_layout(title_text="Dialog vs Narration", template="plotly_white")
fig_seg.write_html(r"C:/Users/Utilisateur/Documents/dialog_narration.html")
fig_seg2.write_html(r"C:/Users/Utilisateur/Documents/dialog_ratio.html")
print("Saved!")


# In[77]:


middlemarch_stats = next(s for s in all_stats if s["title"] == "Bleak House")

# Check top words
print("Top 20 words:")
print(middlemarch_stats["word_freq"].head(20))

# Check total unique words
print("\nTotal unique words:", len(middlemarch_stats["word_freq"]))
print("Max frequency:",       middlemarch_stats["word_freq"]["count"].max())
print("Min frequency:",       middlemarch_stats["word_freq"]["count"].min())

# Check tokens
print("\nSentence count:", middlemarch_stats["n_sentences"])
print("Word count:",      middlemarch_stats["n_words"])

