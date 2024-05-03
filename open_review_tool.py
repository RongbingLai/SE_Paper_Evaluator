import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from papermage.recipes import CoreRecipe

path = "/Users/crystalalice/Desktop/ICSHP_Research/SE_paper/Software_Documentation_Issues_Unveiled.pdf"

def _find_similiar_paper(paper_abstract: list) -> int:
    db_paper_abstracts = []

    with open('correct_file.json', 'r') as file:
        data = json.load(file)
        for submission in data['orb_submissions']:
            if '0' in submission['article_versions']:
                paper = submission['article_versions']['0'] # only need one version
                if paper['title'] and paper['abstract']:
                    db_paper_abstracts.append(paper['title'] + ": " + paper['abstract'])
                    print(paper['title'] + ": " + paper['abstract'] + "\n\n")
                else:
                    db_paper_abstracts.append('')
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_db = tfidf_vectorizer.fit_transform(db_paper_abstracts)
    tfidf_matrix = tfidf_vectorizer.transform(paper_abstract)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix_db).flatten()

    print("Cosine similarity scores:", cosine_sim)

    index = cosine_sim.argmax()
    print("index is " + str(index) + "\n")
    print(db_paper_abstracts[index])

    return index

def _get_paper_abstract() -> list:
    recipe = CoreRecipe()
    doc = recipe.run(path)
    return [doc.titles[0].text + ": " + doc.abstracts[0].text]
    
def get_openreview_review():
    paper_abstract = _get_paper_abstract()
    index = _find_similiar_paper(paper_abstract)

    with open('correct_file.json', 'r') as file:
        data = json.load(file)
        for paper in data['orb_submissions'][index]['article_versions'].values():
            print("title is " + paper['title'] + "\n")
            return paper['reviews'] if len(paper['review']) > 0 else "No Similar Review"


print(get_openreview_review())