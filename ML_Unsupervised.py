{
    "metadata": {
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3 (ipykernel)",
            "language": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 1,
    "cells": [
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Assignment 6 - Implementing k-means with Text Data\n",
                "\n",
                "In this assignment you will implement the k-means algorithm. Unlike past homework assignments, you are not using a pre-implemented sklearn class; you will be implementing the code for k-means yourself.  The assignment uses the `numpy` library for manipulating the data arrays. See the following tutorial for more information on `numpy`.\n",
                "\n",
                "* **[NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)**\n",
                "\n",
                "We will be focusing on the specific setting of clustering text documents, but your k-means implementation will be general to any setting. When properly executed, clustering uncovers valuable insights from a set of unlabeled documents.\n",
                "\n",
                "In this assignment, you will:\n",
                "\n",
                "* Cluster Wikipedia documents using k-means\n",
                "* Explore the role of random initialization on the quality of the clustering\n",
                "* Explore how results differ after changing the number of clusters\n",
                "* Evaluate clustering, both quantitatively and qualitatively\n",
                "\n",
                "Fill in the cells provided marked `TODO` with code to answer the questions. \n",
                "\n",
                "\u003e Copyright ©2023 Emily Fox and Hunter Schafer.  All rights reserved.  Permission is hereby granted to students registered for University of Washington CSE/STAT 416 for use solely during Spring Quarter 2024 for purposes of the course.  No other use, copying, distribution, or modification is permitted without prior written consent. Copyrights for third-party components of this work must be honored.  Instructors interested in reusing these course materials should contact the author.\n",
                "\n",
                "---"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 21,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "\n",
                "%matplotlib inline"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Load data, Extract features"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 22,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/html": "\u003cdiv\u003e\n\u003cstyle scoped\u003e\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n\u003c/style\u003e\n\u003ctable border=\"1\" class=\"dataframe\"\u003e\n  \u003cthead\u003e\n    \u003ctr style=\"text-align: right;\"\u003e\n      \u003cth\u003e\u003c/th\u003e\n      \u003cth\u003eURI\u003c/th\u003e\n      \u003cth\u003ename\u003c/th\u003e\n      \u003cth\u003etext\u003c/th\u003e\n    \u003c/tr\u003e\n  \u003c/thead\u003e\n  \u003ctbody\u003e\n    \u003ctr\u003e\n      \u003cth\u003e0\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Mauno_J%C3%A4rvel...\u003c/td\u003e\n      \u003ctd\u003eMauno J%C3%A4rvel%C3%A4\u003c/td\u003e\n      \u003ctd\u003emauno jrvel born 25 november 1949 in kaustinen...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e1\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/David_W._Jourdan\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eDavid W. Jourdan\u003c/td\u003e\n      \u003ctd\u003edavid walter jourdan born december 5 1954 is a...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e2\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Patrick_Roach\u0026gt;\u003c/td\u003e\n      \u003ctd\u003ePatrick Roach\u003c/td\u003e\n      \u003ctd\u003epatrick roach born march 4 1969 is a canadian ...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e3\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Louis_Sauer\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eLouis Sauer\u003c/td\u003e\n      \u003ctd\u003elouis lou sauer aka louis edward sauer born 19...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e4\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Marty_Keough\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eMarty Keough\u003c/td\u003e\n      \u003ctd\u003erichard martin keough born april 14 1934 in oa...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e5\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Andrea_Jung\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eAndrea Jung\u003c/td\u003e\n      \u003ctd\u003eandrea jung pinyin zhng bnxin jyutping zung1 b...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e6\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Olivia_Mitchell\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eOlivia Mitchell\u003c/td\u003e\n      \u003ctd\u003eolivia mitchell born 31 july 1947 is an irish ...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e7\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Prayut_Chan-o-cha\u0026gt;\u003c/td\u003e\n      \u003ctd\u003ePrayut Chan-o-cha\u003c/td\u003e\n      \u003ctd\u003eprayut chanocha previously spelt prayuth chano...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e8\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Ritchie_Humphreys\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eRitchie Humphreys\u003c/td\u003e\n      \u003ctd\u003eritchie john humphreys born 30 november 1977 i...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e9\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Francisco_G._Ciga...\u003c/td\u003e\n      \u003ctd\u003eFrancisco G. Cigarroa\u003c/td\u003e\n      \u003ctd\u003efrancisco gonzalez cigarroa born december 1 19...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e10\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Marian_Vanghelie\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eMarian Vanghelie\u003c/td\u003e\n      \u003ctd\u003edaniel marian vanghelie b 1968 bucharest is a ...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e11\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Peter_Michael_Hamel\u0026gt;\u003c/td\u003e\n      \u003ctd\u003ePeter Michael Hamel\u003c/td\u003e\n      \u003ctd\u003epeter michael hamel born in munich 15 july 194...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e12\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Paul_Danblon\u0026gt;\u003c/td\u003e\n      \u003ctd\u003ePaul Danblon\u003c/td\u003e\n      \u003ctd\u003epaul danblon born 25 july 1931 is a belgian co...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e13\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Lloyd_McGuire\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eLloyd McGuire\u003c/td\u003e\n      \u003ctd\u003elloyd mcguire is an english actor who has appe...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e14\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Jack_Cressend\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eJack Cressend\u003c/td\u003e\n      \u003ctd\u003ejohn baptiste jack cressend iii born may 13 19...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e15\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Thorsten_Engelmann\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eThorsten Engelmann\u003c/td\u003e\n      \u003ctd\u003ethorsten engelmann born 20 july 1981 in berlin...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e16\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Yelena_Yemchuk\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eYelena Yemchuk\u003c/td\u003e\n      \u003ctd\u003eyelena yemchuk ukrainian born april 22 1970 is...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e17\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Grace_Knight\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eGrace Knight\u003c/td\u003e\n      \u003ctd\u003egrace ethel knight born 23 december 1955 manch...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e18\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Chris_Hadfield\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eChris Hadfield\u003c/td\u003e\n      \u003ctd\u003echris austin hadfield oc oont msc cd born 29 a...\u003c/td\u003e\n    \u003c/tr\u003e\n    \u003ctr\u003e\n      \u003cth\u003e19\u003c/th\u003e\n      \u003ctd\u003e\u0026lt;http://dbpedia.org/resource/Diederik_Hol\u0026gt;\u003c/td\u003e\n      \u003ctd\u003eDiederik Hol\u003c/td\u003e\n      \u003ctd\u003ediederik hol born 10 april 1972 is a dutch des...\u003c/td\u003e\n    \u003c/tr\u003e\n  \u003c/tbody\u003e\n\u003c/table\u003e\n\u003c/div\u003e",
                        "text/plain": "                                                  URI  \\\n0   \u003chttp://dbpedia.org/resource/Mauno_J%C3%A4rvel...   \n1      \u003chttp://dbpedia.org/resource/David_W._Jourdan\u003e   \n2         \u003chttp://dbpedia.org/resource/Patrick_Roach\u003e   \n3           \u003chttp://dbpedia.org/resource/Louis_Sauer\u003e   \n4          \u003chttp://dbpedia.org/resource/Marty_Keough\u003e   \n5           \u003chttp://dbpedia.org/resource/Andrea_Jung\u003e   \n6       \u003chttp://dbpedia.org/resource/Olivia_Mitchell\u003e   \n7     \u003chttp://dbpedia.org/resource/Prayut_Chan-o-cha\u003e   \n8     \u003chttp://dbpedia.org/resource/Ritchie_Humphreys\u003e   \n9   \u003chttp://dbpedia.org/resource/Francisco_G._Ciga...   \n10     \u003chttp://dbpedia.org/resource/Marian_Vanghelie\u003e   \n11  \u003chttp://dbpedia.org/resource/Peter_Michael_Hamel\u003e   \n12         \u003chttp://dbpedia.org/resource/Paul_Danblon\u003e   \n13        \u003chttp://dbpedia.org/resource/Lloyd_McGuire\u003e   \n14        \u003chttp://dbpedia.org/resource/Jack_Cressend\u003e   \n15   \u003chttp://dbpedia.org/resource/Thorsten_Engelmann\u003e   \n16       \u003chttp://dbpedia.org/resource/Yelena_Yemchuk\u003e   \n17         \u003chttp://dbpedia.org/resource/Grace_Knight\u003e   \n18       \u003chttp://dbpedia.org/resource/Chris_Hadfield\u003e   \n19         \u003chttp://dbpedia.org/resource/Diederik_Hol\u003e   \n\n                       name                                               text  \n0   Mauno J%C3%A4rvel%C3%A4  mauno jrvel born 25 november 1949 in kaustinen...  \n1          David W. Jourdan  david walter jourdan born december 5 1954 is a...  \n2             Patrick Roach  patrick roach born march 4 1969 is a canadian ...  \n3               Louis Sauer  louis lou sauer aka louis edward sauer born 19...  \n4              Marty Keough  richard martin keough born april 14 1934 in oa...  \n5               Andrea Jung  andrea jung pinyin zhng bnxin jyutping zung1 b...  \n6           Olivia Mitchell  olivia mitchell born 31 july 1947 is an irish ...  \n7         Prayut Chan-o-cha  prayut chanocha previously spelt prayuth chano...  \n8         Ritchie Humphreys  ritchie john humphreys born 30 november 1977 i...  \n9     Francisco G. Cigarroa  francisco gonzalez cigarroa born december 1 19...  \n10         Marian Vanghelie  daniel marian vanghelie b 1968 bucharest is a ...  \n11      Peter Michael Hamel  peter michael hamel born in munich 15 july 194...  \n12             Paul Danblon  paul danblon born 25 july 1931 is a belgian co...  \n13            Lloyd McGuire  lloyd mcguire is an english actor who has appe...  \n14            Jack Cressend  john baptiste jack cressend iii born may 13 19...  \n15       Thorsten Engelmann  thorsten engelmann born 20 july 1981 in berlin...  \n16           Yelena Yemchuk  yelena yemchuk ukrainian born april 22 1970 is...  \n17             Grace Knight  grace ethel knight born 23 december 1955 manch...  \n18           Chris Hadfield  chris austin hadfield oc oont msc cd born 29 a...  \n19             Diederik Hol  diederik hol born 10 april 1972 is a dutch des...  "
                    },
                    "execution_count": 22,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "### SKIP\n",
                "wiki = pd.read_csv('people_wiki.csv')\n",
                "wiki.head(20)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 23,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_load_data) ###"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "To work with text data, we must first convert the documents into numerical features. Let's extract TF-IDF features for each article."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 24,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "array(['00', '000', '0001', ..., 'zyzzyva', 'zz', 'zzap64'], dtype=object)"
                    },
                    "execution_count": 24,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "from sklearn.feature_extraction.text import TfidfVectorizer\n",
                "\n",
                "vectorizer = TfidfVectorizer(max_df=0.95)  # ignore words with very high doc frequency\n",
                "tf_idf = vectorizer.fit_transform(wiki['text'])\n",
                "words = vectorizer.get_feature_names_out()\n",
                "words"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Since most documents don't contain every word, many of the TF-IDF entries will be 0. Representing the TF-IDF matrix as a `numpy` matrix will require a lot of unnecessary storage to keep track of all those 0. `scikit-learn` instead represents it as a SciPy \"sparse matrix\" that only represents the non-zero entries of a matrix to save space. Externally, you treat it just like a numpy `matrix` but it takes up less storage."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The above matrix contains a TF-IDF score for each of the 5907 pages in the data set and each of the 112801 unique words. You can treat it like any matrix of data but internally it is space-efficient since it knows not to store the zero word counts."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Normalize all vectors"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Euclidean distance can be a poor metric of similarity between documents, since it is very sensitive to the length of an article. As an example, we want long and short articles about baseball players to be treated as similar, but Euclidean distance would treat them as very different due to the length. For a better assessment of similarity, we should disregard the length information and use length-agnostic metrics, such as cosine distance. Unfortunately, the k-means algorithm doesn't work with non-Euclidean distance metrics.\n",
                "\n",
                "We take an alternative route to remove length information: we normalize all vectors to be unit length. It turns out that Euclidean distance closely mimics cosine distance when all vectors are unit length. In particular, the squared Euclidean distance between any two vectors of length one is directly proportional to their cosine distance.\n",
                "\n",
                "---\n",
                "\n",
                "### Optional: Justification\n",
                "This section has some optional background material as to why normalizing makes sense here. You can skip down to the next line break if you don't want to read this.\n",
                "\n",
                "We can prove this as follows. Let $\\mathbf{x}$ and $\\mathbf{y}$ be normalized vectors, i.e. unit vectors, so that $\\|\\mathbf{x}\\|=\\|\\mathbf{y}\\|=1$. Write the squared Euclidean distance as the dot product of $(\\mathbf{x} - \\mathbf{y})$ to itself:\n",
                "\n",
                "$$\\begin{align*}\n",
                "\\|\\mathbf{x} - \\mathbf{y}\\|_2^2 \u0026= (\\mathbf{x} - \\mathbf{y})^T(\\mathbf{x} - \\mathbf{y}) \u0026 \\text{(def of L2 norm)}\\\\\n",
                "                              \u0026= (\\mathbf{x}^T \\mathbf{x}) - 2(\\mathbf{x}^T \\mathbf{y}) + (\\mathbf{y}^T \\mathbf{y}) \u0026 \\text{(FOIL expression)}\\\\\n",
                "                              \u0026= \\|\\mathbf{x}\\|_2^2 - 2(\\mathbf{x}^T \\mathbf{y}) + \\|\\mathbf{y}\\|_2^2 \u0026 \\text{(def of L2 norm)}\\\\\n",
                "                              \u0026= 2 - 2(\\mathbf{x}^T \\mathbf{y}) \u0026 \\text{($\\mathbf{x}$ and $\\mathbf{y}$ are length 1)}\\\\\n",
                "                              \u0026= 2(1 - (\\mathbf{x}^T \\mathbf{y}))\\\\\n",
                "                              \u0026= 2\\left(1 - \\frac{\\mathbf{x}^T \\mathbf{y}}{\\|\\mathbf{x}\\|_2\\|\\mathbf{y}\\|_2}\\right) \u0026 \\text{(Dividing by 1 doesn't change value)}\\\\\n",
                "                              \u0026= 2 \\cdot cosine\\_distance(\\mathbf{x}, \\mathbf{y})\n",
                "\\end{align*}$$\n",
                "\n",
                "This tells us that two **unit vectors** that are close in Euclidean distance are also close in cosine distance. Thus, the k-means algorithm (which naturally uses Euclidean distances) on normalized vectors will produce the same results as clustering using cosine distance as a distance metric.\n",
                "\n",
                "*End optional section*.\n",
                "\n",
                "---\n",
                "We import the [`normalize()` function](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html) from scikit-learn to normalize all vectors to unit length."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 25,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "\u003c5907x112801 sparse matrix of type '\u003cclass 'numpy.float64'\u003e'\n\twith 993501 stored elements in Compressed Sparse Row format\u003e"
                    },
                    "execution_count": 25,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "from sklearn.preprocessing import normalize\n",
                "tf_idf = normalize(tf_idf)\n",
                "tf_idf"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Implement k-means"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The bulk of this assignment will be implementing k-means. We will tackle it in parts to make it manageable.\n",
                "\n",
                "First, we choose an initial set of centroids. A common practice is to choose randomly from the data points.\n",
                "\n",
                "**Note:** We specify a seed here, so that everyone gets the same answer. In practice, we highly recommend to use different seeds every time (for instance, by using the current timestamp)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 26,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_initial_centroids(data, k, seed=None):\n",
                "    \"\"\"\n",
                "    Randomly choose k data points as initial centroids\n",
                "    \"\"\"\n",
                "    if seed is not None: # useful for obtaining consistent results\n",
                "        np.random.seed(seed)\n",
                "        \n",
                "    n = data.shape[0] # number of data points\n",
                "        \n",
                "    # Pick K indices from range [0, n) without replacement.\n",
                "    rand_indices = np.random.choice(n, k)\n",
                "    \n",
                "    # Keep centroids as dense (not sparse) matrix format, as many entries \n",
                "    # will be nonzero due to averaging. As long as at least one document \n",
                "    # in a cluster contains a word, it will carry a nonzero weight in the \n",
                "    # TF-IDF vector of the centroid.\n",
                "    centroids = data[rand_indices,:].toarray()\n",
                "    \n",
                "    return centroids\n",
                ""
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### k-means Algorithm\n",
                "After initialization, the k-means algorithm iterates between the following two steps:\n",
                "1. Assign each data point to the closest centroid. $$z_i \\gets \\mathrm{argmin}_{j\\in[k]} \\|\\mathbf{\\mu}_j - \\mathbf{x}_i\\|_2^2$$\n",
                "2. Revise centroids to be the mean of the assigned data points. $$\\mu^{(j)} \\gets \\frac{\\sum_{i=1}^n \\mathbb{1}\\{z_i = j\\}x_i}{\\sum_{i=1}^n \\mathbb{1}\\{z_i = j\\}}$$"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "In pseudocode, we iteratively do the following:\n",
                "```python\n",
                "cluster_assignment = assign_clusters(data, centroids)\n",
                "centroids = revise_centroids(data, k, cluster_assignment)\n",
                "```"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Assigning clusters"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "How do we implement Step 1 of the main k-means loop above? First we import `pairwise_distances` function from scikit-learn, which calculates Euclidean distances between rows of given arrays. See [this documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html) for more information.\n",
                "\n",
                "For the sake of demonstration, let's look at documents 100 through 102 as query documents and compute the distances between each of these documents and every other document in the corpus. In the k-means algorithm, we will have to compute pairwise distances between the set of centroids and the set of documents."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 27,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "[[1.39996239 1.39958932]\n [1.40386156 1.39754968]\n [1.38421176 1.39682604]\n ...\n [1.40562888 1.39024794]\n [1.39673862 1.38306708]\n [1.40872806 1.40250208]]\n"
                }
            ],
            "source": [
                "from sklearn.metrics import pairwise_distances\n",
                "\n",
                "# Get the TF-IDF vectors for documents 100 through 102.\n",
                "queries = tf_idf[100:102,:]\n",
                "\n",
                "# Compute pairwise distances from every data point to each query vector.\n",
                "dist = pairwise_distances(tf_idf, queries, metric='euclidean')\n",
                "print(dist)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "More formally, `dist[i,j]` is assigned the distance between the `i`th row of `X` (i.e., `X[i,:]`) and the `j`th row of `Y` (i.e., `Y[j,:]`)."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 1** Computing Distances\n",
                "\n",
                "To test your understanding of how this code works, in the cell below write code that does the following tasks\n",
                "\n",
                "* Initializes 3 centroids that are the first 3 rows of `tf_idf`\n",
                "* Compute the distances between all the points in `tf_idf` and those 3 centroids. The result should be a matrix with shape `(5907, 3)`. Store this in a variable called `distances`.\n",
                "* Use `distances` to find the distance between the row of `tf_idf` with index 430 to the second centroid (index 1). Store this value in a variable called `dist`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 28,
            "metadata": {},
            "outputs": [
                {
                    "ename": "IndexError",
                    "evalue": "invalid number of indices",
                    "output_type": "error",
                    "traceback": [
                        "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[0;31mIndexError\u001b[0m                                Traceback (most recent call last)",
                        "Cell \u001b[0;32mIn[1], line 9\u001b[0m\n\u001b[1;32m      5\u001b[0m centroids \u001b[38;5;241m=\u001b[39m tf_idf[:\u001b[38;5;241m3\u001b[39m]\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# Step 2: Compute the distances between all the points in tf_idf and the centroids\u001b[39;00m\n\u001b[1;32m      8\u001b[0m \u001b[38;5;66;03m# We'll use numpy broadcasting to compute the distances efficiently\u001b[39;00m\n\u001b[0;32m----\u003e 9\u001b[0m distances \u001b[38;5;241m=\u001b[39m np\u001b[38;5;241m.\u001b[39msqrt(((\u001b[43mtf_idf\u001b[49m\u001b[43m[\u001b[49m\u001b[43m:\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mnewaxis\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43m:\u001b[49m\u001b[43m]\u001b[49m \u001b[38;5;241m-\u001b[39m centroids) \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39m \u001b[38;5;241m2\u001b[39m)\u001b[38;5;241m.\u001b[39msum(axis\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m2\u001b[39m))\n\u001b[1;32m     11\u001b[0m \u001b[38;5;66;03m# Step 3: The result should be a matrix with shape (5907, 3)\u001b[39;00m\n\u001b[1;32m     12\u001b[0m \u001b[38;5;66;03m# distances variable already stores the required matrix\u001b[39;00m\n\u001b[1;32m     13\u001b[0m \n\u001b[1;32m     14\u001b[0m \u001b[38;5;66;03m# Step 4: Use distances to find the distance between the row of tf_idf with index 430 to the second centroid (index 1)\u001b[39;00m\n\u001b[1;32m     15\u001b[0m dist \u001b[38;5;241m=\u001b[39m distances[\u001b[38;5;241m430\u001b[39m, \u001b[38;5;241m1\u001b[39m]\n",
                        "File \u001b[0;32m/usr/lib/python3.11/site-packages/scipy/sparse/_index.py:52\u001b[0m, in \u001b[0;36mIndexMixin.__getitem__\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m     51\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m__getitem__\u001b[39m(\u001b[38;5;28mself\u001b[39m, key):\n\u001b[0;32m---\u003e 52\u001b[0m     row, col \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_validate_indices\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     54\u001b[0m     \u001b[38;5;66;03m# Dispatch to specialized methods.\u001b[39;00m\n\u001b[1;32m     55\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(row, INT_TYPES):\n",
                        "File \u001b[0;32m/usr/lib/python3.11/site-packages/scipy/sparse/_index.py:162\u001b[0m, in \u001b[0;36mIndexMixin._validate_indices\u001b[0;34m(self, key)\u001b[0m\n\u001b[1;32m    160\u001b[0m     row, col \u001b[38;5;241m=\u001b[39m key\u001b[38;5;241m.\u001b[39mnonzero()\n\u001b[1;32m    161\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--\u003e 162\u001b[0m     row, col \u001b[38;5;241m=\u001b[39m \u001b[43m_unpack_index\u001b[49m\u001b[43m(\u001b[49m\u001b[43mkey\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    163\u001b[0m M, N \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mshape\n\u001b[1;32m    165\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_validate_bool_idx\u001b[39m(\n\u001b[1;32m    166\u001b[0m     idx: npt\u001b[38;5;241m.\u001b[39mNDArray[np\u001b[38;5;241m.\u001b[39mbool_],\n\u001b[1;32m    167\u001b[0m     axis_size: \u001b[38;5;28mint\u001b[39m,\n\u001b[1;32m    168\u001b[0m     axis_name: \u001b[38;5;28mstr\u001b[39m\n\u001b[1;32m    169\u001b[0m ) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m\u003e\u001b[39m npt\u001b[38;5;241m.\u001b[39mNDArray[np\u001b[38;5;241m.\u001b[39mint_]:\n",
                        "File \u001b[0;32m/usr/lib/python3.11/site-packages/scipy/sparse/_index.py:313\u001b[0m, in \u001b[0;36m_unpack_index\u001b[0;34m(index)\u001b[0m\n\u001b[1;32m    311\u001b[0m         row, col \u001b[38;5;241m=\u001b[39m index[\u001b[38;5;241m0\u001b[39m], \u001b[38;5;28mslice\u001b[39m(\u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[1;32m    312\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--\u003e 313\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mIndexError\u001b[39;00m(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124minvalid number of indices\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m    314\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    315\u001b[0m     idx \u001b[38;5;241m=\u001b[39m _compatible_boolean_index(index)\n",
                        "\u001b[0;31mIndexError\u001b[0m: invalid number of indices"
                    ]
                }
            ],
            "source": [
                "### edTest(test_q1_computing_distances) ###\n",
                "\n",
                "# TODO Fill out this cell\n",
                "# Step 1: Initialize 3 centroids that are the first 3 rows of tf_idf\n",
                "centroids = tf_idf[:3]\n",
                "\n",
                "# Step 2: Compute the distances between all the points in tf_idf and the centroids\n",
                "distances = np.sqrt(((tf_idf[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))\n",
                "\n",
                "# Step 3: The result should be a matrix with shape (5907, 3)\n",
                "# distances variable already stores the required matrix\n",
                "\n",
                "# Step 4: Use distances to find the distance between the row of tf_idf with index 430 to the second centroid (index 1)\n",
                "dist = distances[430, 1]\n",
                "\n",
                "distances =  ...\n",
                "dist = ..."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 2** Find closest centroid\n",
                "\n",
                "Now that you have computed the pairwise distances between all datapoints in `tf_idf` and the 3 centroids, your task is to find the closest centroid for each datapoint and store it in the variable `closest_cluster`. To do so, for each datapoint find the the centroid with the minimum distance to the datapoint. Fittingly, NumPy provides an `argmin` function. See [this documentation](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.argmin.html) for details. \n",
                "\n",
                "`closest_cluster` should be a 1D array whose $i^{th}$ entry contains the index of the centroid that is the closest to the $i^{th}$ datapoint. Note that it would be **unreasonably slow** run this code if you use a for/while loop; therefore, you **must use [argmin](http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.argmin.html)**. (See documentation for examples)\n",
                "\n",
                "*Hint:* the resulting array should be as long as the number of data points. Think carefully about which axis you want to take the argmin on to get the index of the centroid with the minimum distance to that datapoint."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q2_closest_centroid) ###\n",
                "\n",
                "# TODO Fill out this cell\n",
                "closest_cluster = ..."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 3** Assign Clusters\n",
                "\n",
                "Now that we have completed components of the Step 1 code, let's put it together in a single function that takes a dataset and centroids and assigns each row to the closest centroid. We are ready to fill in the blanks in the aforementioned function `assign_clusters(data, centroids)`. "
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q3_assign_clusters) ###\n",
                "\n",
                "# TODO Complete this function\n",
                "def assign_clusters(data, centroids):\n",
                "    \"\"\"\n",
                "    Parameters:  \n",
                "      - data      - is an np.array of float values with n rows and d columns.  \n",
                "      - centroids - is an np.array of float values with k rows and dcolumns.\n",
                "\n",
                "    Returns  \n",
                "      -  A np.array of length n where the ith index represents which centroid \n",
                "         data[i] was assigned to. The assignments range between the values 0, ..., k-1.\n",
                "    \"\"\"\n",
                "    # TODO get the distance between data and centroids\n",
                "    \n",
                "    # TODO get the closest centroid for each datapoint\n",
                "\n",
                "    # TODO return an array that contains the closest centroid index for each datapoint\n",
                "    return ...\n",
                "    "
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Revising clusters\n",
                "\n",
                "### Numpy Tutorial\n",
                "Let's turn to Step 2 of the k-means algorithm, where we compute the new centroids given the current cluster assignments. \n",
                "\n",
                "SciPy and NumPy arrays allow for filtering via Boolean masks. For instance, we filter all data points that are assigned to cluster 0 by writing\n",
                "\n",
                "```python\n",
                "data[cluster_assignment == 0, :]\n",
                "```\n",
                "\n",
                "To develop intuition about filtering, let's look at a small example consisting of 3 data points and 2 clusters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {},
            "outputs": [],
            "source": [
                "data = np.array([[1., 2., 0.],\n",
                "                 [0., 0., 0.],\n",
                "                 [2., 2., 0.]])\n",
                "centroids = np.array([[0.5, 0.5, 0.],\n",
                "                      [0., -0.5, 0.]])"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's assign these data points to the closest centroid using the function you wrote before."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 12,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "Ellipsis"
                    },
                    "execution_count": 12,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "cluster_assignment = assign_clusters(data, centroids)\n",
                "cluster_assignment"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "The expression `cluster_assignment == 1` gives a list of Booleans that says whether each data point is assigned to cluster 1 or not:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 13,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "False"
                    },
                    "execution_count": 13,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "cluster_assignment == 1"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Likewise for cluster 0:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 14,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "False"
                    },
                    "execution_count": 14,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "cluster_assignment == 0"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "When indexing into a numpy array, instead of indices, we can also put in the list of Booleans to pick and choose rows. Only the rows that correspond to a `True` entry will be retained.\n",
                "\n",
                "First, let's look at the values of the data points that were assigned to cluster 1:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 15,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "array([], shape=(0, 3, 3), dtype=float64)"
                    },
                    "execution_count": 15,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "data[cluster_assignment == 1]"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "This makes sense since the vector `[0 0 0]` is closer to the centroid `[0 -0.5 0]` than to the centroid `[0.5 0.5 0]`.\n",
                "\n",
                "Now let's look at the data points assigned to cluster 0:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 16,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "text/plain": "array([], shape=(0, 3, 3), dtype=float64)"
                    },
                    "execution_count": 16,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "data[cluster_assignment == 0]"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Again, this makes sense since these values are each closer to the centroid `[0.5 0.5 0]` than to `[0 -0.5 0]`.\n",
                "\n",
                "Given all the data points in a cluster, it only remains to compute the mean. Use [np.mean()](http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.mean.html). By default, the function averages all elements in a 2D array. To compute row-wise or column-wise means, add the `axis` argument. See the linked documentation for details. \n",
                "\n",
                "In the cell below, we first find all the rows that were assigned cluster 0 and then take the average of those vectors to find the new cluster 0 centroid. Notice that the result will be an np.array with 3 elements because that's the dimensionality of the vectors."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 17,
            "metadata": {},
            "outputs": [
                {
                    "name": "stderr",
                    "output_type": "stream",
                    "text": "/tmp/ipykernel_30/1342920855.py:1: RuntimeWarning: Mean of empty slice.\n  data[cluster_assignment==0].mean(axis=0)\n/usr/lib/python3.11/site-packages/numpy/core/_methods.py:121: RuntimeWarning: invalid value encountered in divide\n  ret = um.true_divide(\n"
                },
                {
                    "data": {
                        "text/plain": "array([[nan, nan, nan],\n       [nan, nan, nan],\n       [nan, nan, nan]])"
                    },
                    "execution_count": 17,
                    "metadata": {},
                    "output_type": "execute_result"
                }
            ],
            "source": [
                "data[cluster_assignment==0].mean(axis=0)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 4** Revise Centroids\n",
                "\n",
                "\n",
                "Now we are ready to fill in the blanks in the function `revise_centroids(data, k, cluster_assignment)`. In the cell below, complete the `...` sections to compute the new centroids given the current cluster assignment."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 18,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q4_revise_centroids) ###\n",
                "\n",
                "# TODO fill in this function\n",
                "def revise_centroids(data, k, cluster_assignment):\n",
                "    \"\"\"\n",
                "    Parameters:  \n",
                "      - data               - is an np.array of float values with n rows and d columns.\n",
                "      - k                  - number of centroids\n",
                "      - cluster_assignment - np.array of length n where the ith index represents which \n",
                "                             centroid data[i] was assigned to. The assignments range between the values 0, ..., k-1.\n",
                "\n",
                "    Returns  \n",
                "      -  A np.array with k rows and d columns for the new centroids.\n",
                "    \"\"\"\n",
                "    new_centroids = []\n",
                "    for i in range(k):\n",
                "        # TODO Select all data points that belong to cluster i. Fill in the ... portion\n",
                "        member_data_points = ...\n",
                "        \n",
                "        # TODO Compute the mean of the member data points. Fill in the ... portion\n",
                "        centroid = ...\n",
                "        \n",
                "        # Convert numpy.matrix type to numpy.ndarray type\n",
                "        centroid = centroid.A1\n",
                "        new_centroids.append(centroid)\n",
                "        \n",
                "    new_centroids = np.array(new_centroids)\n",
                "    return new_centroids"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Assessing convergence"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "How can we tell if the k-means algorithm is converging? We can look at the cluster assignments and see if they stabilize over time. In fact, we'll be running the algorithm until the cluster assignments stop changing at all. To be extra safe, and to assess the clustering performance, we'll be looking at an additional criteria: the sum of all squared distances between data points and centroids (called the \"heterogeneity objective\" in lecture). This is defined as\n",
                "$$\n",
                "J(\\mathcal{Z},\\mu) = \\sum_{j=1}^{k} \\sum_{i=1}^n \\mathbb{1}\\{z_i =j\\}\\|x_i - \\mu^{(j)}\\|^2_2.\n",
                "$$\n",
                "The smaller the distances, the more homogeneous the clusters are. In other words, we'd like to have \"tight\" clusters."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 19,
            "metadata": {},
            "outputs": [],
            "source": [
                "def compute_heterogeneity(data, k, centroids, cluster_assignment):\n",
                "    \"\"\"\n",
                "    Computes the heterogeneity metric of the data using the given centroids and cluster assignments.\n",
                "    \"\"\"\n",
                "    heterogeneity = 0.0\n",
                "    for i in range(k):\n",
                "        \n",
                "        # Select all data points that belong to cluster i. Fill in the blank (RHS only)\n",
                "        member_data_points = data[cluster_assignment == i, :]\n",
                "        \n",
                "        if member_data_points.shape[0] \u003e 0: # check if i-th cluster is non-empty\n",
                "            # Compute distances from centroid to data point\n",
                "            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')\n",
                "            squared_distances = distances ** 2\n",
                "            heterogeneity += np.sum(squared_distances)\n",
                "        \n",
                "    return heterogeneity\n",
                "\n",
                "    "
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's compute the cluster heterogeneity for the 2-cluster example we've been considering based on our current cluster assignments and centroids."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 20,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": "0.0\n"
                },
                {
                    "ename": "AttributeError",
                    "evalue": "'ellipsis' object has no attribute 'shape'",
                    "output_type": "error",
                    "traceback": [
                        "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
                        "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
                        "Cell \u001b[0;32mIn[1], line 2\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28mprint\u001b[39m(compute_heterogeneity(data, \u001b[38;5;241m2\u001b[39m, centroids, cluster_assignment))\n\u001b[0;32m----\u003e 2\u001b[0m \u001b[43mcluster_assignment\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mshape\u001b[49m\n",
                        "\u001b[0;31mAttributeError\u001b[0m: 'ellipsis' object has no attribute 'shape'"
                    ]
                }
            ],
            "source": [
                "print(compute_heterogeneity(data, 2, centroids, cluster_assignment))\n",
                "cluster_assignment.shape"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 5** Combining it into a single function"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Once the two k-means steps have been implemented, as well as our heterogeneity metric we wish to monitor, it is only a matter of putting these functions together to write a k-means algorithm that\n",
                "\n",
                "* Repeatedly performs Steps 1 and 2\n",
                "* Tracks convergence metrics on each iteration\n",
                "* Stops if either no assignment changed or we reach a certain number of iterations."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Now we are ready to fill in the blanks the function `kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False)`. In the cell below, complete the `...` sections to meet the specification of the function."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 44,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q5_kmeans) ###\n",
                "\n",
                "# TODO Fill in the blanks\n",
                "def kmeans(data, k, initial_centroids, max_iter, record_heterogeneity=None, verbose=False):\n",
                "    \"\"\"\n",
                "    This function runs k-means on given data and initial set of centroids.\n",
                "    \n",
                "    Parameters:  \n",
                "      - data                 - is an np.array of float values of length N.\n",
                "      - k                    - number of centroids\n",
                "      - initial_centroids    - is an np.array of float values of length k.\n",
                "      - max_iter             - maximum number of iterations to run the algorithm\n",
                "      - record_heterogeneity - if provided an empty list, it will compute the heterogeneity \n",
                "                               at each iteration and append it to the list. \n",
                "                               Defaults to None and won't record heterogeneity.\n",
                "      - verbose              - set to True to display progress. Defaults to False and won't \n",
                "                               display progress.\n",
                "\n",
                "    Returns  \n",
                "      - centroids - A np.array of length k for the centroids upon termination of the algorithm.\n",
                "      - cluster_assignment - A np.array of length n where the ith index represents which \n",
                "                             centroid data[i] was assigned to. The assignments range between the \n",
                "                             values 0, ..., k-1 upon termination of the algorithm.\n",
                "    \"\"\"\n",
                "    centroids = initial_centroids[:]\n",
                "    prev_cluster_assignment = None\n",
                "    \n",
                "    for itr in range(max_iter):  \n",
                "        # Print itereation number\n",
                "        if verbose:\n",
                "            print(itr)\n",
                "        \n",
                "        # TODO 1. Make assign each datapoint to the nearest centroid\n",
                "        cluster_assignment = ...\n",
                "            \n",
                "        # TODO 2. Compute a new centroid for each of the k clusters, by averaging all \n",
                "        # data points assigned to that cluster.\n",
                "        centroids = ...\n",
                "            \n",
                "        # Check for convergence: if none of the assignments changed, stop\n",
                "        if prev_cluster_assignment is not None and \\\n",
                "          (prev_cluster_assignment == cluster_assignment).all():\n",
                "            break\n",
                "        \n",
                "        # Print number of new assignments \n",
                "        if prev_cluster_assignment is not None:\n",
                "            num_changed = sum(abs(prev_cluster_assignment - cluster_assignment))\n",
                "            if verbose:\n",
                "                print(f'    {num_changed:5d} elements changed their cluster assignment.')  \n",
                "        \n",
                "        # Record heterogeneity convergence metric\n",
                "        if record_heterogeneity is not None:\n",
                "            # TODO compute the heterogeneity of the cluster\n",
                "            score = ...\n",
                "            record_heterogeneity.append(score)\n",
                "        \n",
                "        prev_cluster_assignment = cluster_assignment[:]\n",
                "        \n",
                "    return centroids, cluster_assignment"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Plotting convergence metric"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "We can use the above function to plot the convergence metric across iterations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 45,
            "metadata": {},
            "outputs": [],
            "source": [
                "def plot_heterogeneity(heterogeneity, k):\n",
                "    \"\"\"\n",
                "    Plots how the heterogeneity changes as the number of iterations increases.\n",
                "    \"\"\"\n",
                "    plt.figure(figsize=(7,4))\n",
                "    plt.plot(heterogeneity, linewidth=4)\n",
                "    plt.xlabel('# Iterations')\n",
                "    plt.ylabel('Heterogeneity')\n",
                "    plt.title(f'Heterogeneity of clustering over time, K={k}')\n",
                "    plt.rcParams.update({'font.size': 16})\n",
                "    plt.tight_layout()"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's consider running k-means with $k=3$ clusters for a maximum of 400 iterations, recording cluster heterogeneity at every step.  Then, let's plot the heterogeneity over iterations using the plotting function above. We include a seed to ensure everyone gets the same results."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 48,
            "metadata": {},
            "outputs": [],
            "source": [
                "k = 3\n",
                "heterogeneity = []\n",
                "initial_centroids = get_initial_centroids(tf_idf, k, seed=0)\n",
                "centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=400,\n",
                "                                       record_heterogeneity=heterogeneity, verbose=True)\n",
                "plot_heterogeneity(heterogeneity, k)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 6** Largest cluster\n",
                "\n",
                "Using the output of k-means from the last cell, write code to compute which final cluster contains the most data points. Store your result as the index of that cluster (an `int`) in a variable named `largest_cluster`, \n",
                "representing which cluster in `centroids` has the most datapoints assigned to it. You may want to use [np.bincount](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html).\n",
                ""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 50,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q6_largest_cluster) ###\n",
                "\n",
                "# TODO Find largest cluster from example above\n",
                "largest_cluster = ...\n",
                ""
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Beware of Local Minima"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "One weakness of k-means is that it tends to get stuck in a local minimum based on its starting position. To see this, let us run k-means multiple times, with different initial centroids created using different random seeds.\n",
                "\n",
                "**Note:** Again, in practice, you should set different seeds for every run. We give you a list of seeds for this assignment so that everyone gets the same answer.\n",
                "\n",
                "This may take a minute or two to run."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 51,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "# ^ Magic command to time how long it takes for this cell to run!\n",
                "# You can see how long it took with the output that says \"Wall time\"\n",
                "\n",
                "k = 10\n",
                "heterogeneity = {}\n",
                "for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:\n",
                "    initial_centroids = get_initial_centroids(tf_idf, k, seed)\n",
                "    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=800,\n",
                "                                           record_heterogeneity=None, verbose=False)\n",
                "    # To save time, compute heterogeneity only once in the end\n",
                "    heterogeneity[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)\n",
                "    print(f'seed={seed:06d}, heterogeneity={heterogeneity[seed]:.5f}')\n",
                ""
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Notice the variation in heterogeneity for different initializations. This indicates that k-means runs may have not converged or they got stuck at a local minimum.\n",
                "\n",
                "# k-means++\n",
                "One effective way to counter this tendency is to use **k-means++** to provide a smart initialization. This method tries to spread out the initial set of centroids so that they are not too close together. It is known to improve the quality of local optima and lower average runtime, but is a bit slower to start since it needs to do more computation to place centroids."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 52,
            "metadata": {},
            "outputs": [],
            "source": [
                "def k_means_plus_plus_initialization(data, k, seed=None):\n",
                "    \"\"\"\n",
                "    Use k-means++ to initialize a good set of centroids\n",
                "    \"\"\"\n",
                "    if seed is not None: # useful for obtaining consistent results\n",
                "        np.random.seed(seed)\n",
                "        \n",
                "    centroids = np.zeros((k, data.shape[1]))\n",
                "    \n",
                "    # Randomly choose the first centroid.\n",
                "    # Since we have no prior knowledge, choose uniformly at random\n",
                "    idx = np.random.randint(data.shape[0])\n",
                "    centroids[0] = data[idx,:].toarray()\n",
                "    \n",
                "    # Compute distances from the first centroid chosen to all the other data points\n",
                "    distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten()\n",
                "    \n",
                "    for i in range(1, k):\n",
                "        # Choose the next centroid randomly, so that the probability for each data point to be chosen\n",
                "        # is directly proportional to its squared distance from the nearest centroid.\n",
                "        # Roughtly speaking, a new centroid should be as far as from ohter centroids as possible.\n",
                "        idx = np.random.choice(data.shape[0], 1, p=distances/sum(distances))\n",
                "        centroids[i] = data[idx,:].toarray()\n",
                "        \n",
                "        # Now compute distances from the centroids to all data points\n",
                "        distances = np.min(pairwise_distances(data, centroids[0:i+1], metric='euclidean'),axis=1)\n",
                "    \n",
                "    return centroids"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's now rerun k-means with 10 clusters using the same set of seeds, but always using k-means++ to initialize the algorithm.\n",
                "\n",
                "This may take several minutes to run."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 53,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "\n",
                "k = 10\n",
                "heterogeneity_smart = {}\n",
                "for seed in [0, 20000, 40000, 60000, 80000, 100000, 120000]:\n",
                "    initial_centroids = k_means_plus_plus_initialization(tf_idf, k, seed)\n",
                "    centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, max_iter=400,\n",
                "                                           record_heterogeneity=None, verbose=False)\n",
                "    # To save time, compute heterogeneity only once in the end\n",
                "    heterogeneity_smart[seed] = compute_heterogeneity(tf_idf, k, centroids, cluster_assignment)\n",
                "    print(f'seed={seed:06d}, heterogeneity={heterogeneity_smart[seed]:.5f}')"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let's compare the set of cluster heterogeneities we got from our 7 restarts of k-means using random initialization compared to the 7 restarts of k-means using k-means++ as a smart initialization.\n",
                "\n",
                "The following code produces a [box plot](http://matplotlib.org/api/pyplot_api.html) for each of these methods, indicating the spread of heterogeneity values produced by each method."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 54,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(10,5))\n",
                "plt.boxplot([list(heterogeneity.values()), list(heterogeneity_smart.values())], vert=False)\n",
                "\n",
                "plt.yticks([1, 2], ['k-means', 'k-means++'])\n",
                "plt.xlabel('Heterogeneity')\n",
                "plt.title('Heterogeneity distribution of k-means vs k-means++')\n",
                "\n",
                "plt.rcParams.update({'font.size': 16})\n",
                "plt.tight_layout()"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "A few things to notice from the box plot:\n",
                "* Random initialization results in a worse clustering than k-means++ on average.\n",
                "* The best result of k-means++ is better than the best result of random initialization.\n",
                "\n",
                "## 🔍 **Question 7** kmeans_multiple_runs\n",
                "**In general, you should run k-means at least a few times with different initializations and then return the run resulting in the lowest heterogeneity.** Let us write a function that runs k-means multiple times and picks the best run that minimizes heterogeneity.\n",
                "\n",
                "Now we are ready to fill in the blanks the function `kmeans_multiple_runs(data, k, max_iter, verbose=False)`. \n",
                "\n",
                ""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 55,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q7_kmeans_multiple_runs) ###\n",
                "\n",
                "# TODO Fill in the ...\n",
                "def kmeans_multiple_runs(data, k, max_iter, seeds, verbose=False):\n",
                "    \"\"\"\n",
                "    Runs kmeans multiple times \n",
                "    \n",
                "    Parameters:  \n",
                "      - data     - is an np.array of float values of length n.\n",
                "      - k        - number of centroids\n",
                "      - max_iter - maximum number of iterations to run the algorithm\n",
                "      - seeds    - Either number of seeds to try (generated randomly) or a list of seed values\n",
                "      - verbose  - set to True to display progress. Defaults to False and won't display progress.\n",
                "    \n",
                "    Returns  \n",
                "      - final_centroids          - A np.array of length k for the centroids upon \n",
                "                                   termination of the algorithm.\n",
                "      - final_cluster_assignment - A np.array of length n where the ith index represents which \n",
                "                                   centroid data[i] was assigned to. The assignments range between \n",
                "                                   the values 0, ..., k-1 upon termination of the algorithm.\n",
                "    \"\"\"    \n",
                "    min_heterogeneity_achieved = float('inf')\n",
                "    final_centroids = None\n",
                "    final_cluster_assignment = None\n",
                "    if type(seeds) == int:\n",
                "        seeds = np.random.randint(low=0, high=10000, size=seeds)\n",
                "    \n",
                "    num_runs = len(seeds)\n",
                "    \n",
                "    for seed in seeds:\n",
                "        # TODO Use k-means++ initialization: Fill in the blank\n",
                "        # Set record_heterogeneity=None because we will compute that once at the end.\n",
                "        initial_centroids = ...\n",
                "\n",
                "        # TODO Run k-means: Fill in the blank \n",
                "        centroids, cluster_assignment = ...\n",
                "        \n",
                "        # TODO To save time, compute heterogeneity only once in the end\n",
                "        seed_heterogeneity = ...\n",
                "        \n",
                "        if verbose:\n",
                "            print(f'seed={seed:06d}, heterogeneity={seed_heterogeneity:.5f}')\n",
                "        \n",
                "        # if current measurement of heterogeneity is lower than previously seen,\n",
                "        # update the minimum record of heterogeneity.\n",
                "        if seed_heterogeneity \u003c min_heterogeneity_achieved:\n",
                "            min_heterogeneity_achieved = seed_heterogeneity\n",
                "            final_centroids = centroids\n",
                "            final_cluster_assignment = cluster_assignment\n",
                "    \n",
                "    # Return the centroids and cluster assignments that minimize heterogeneity.\n",
                "    return final_centroids, final_cluster_assignment"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## How to choose K"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Since we are measuring the tightness of the clusters, a higher value of K reduces the possible heterogeneity metric by definition.  For example, if we have N data points and set K=N clusters, then we could have 0 cluster heterogeneity by setting the N centroids equal to the values of the N data points. (Note: Not all runs for larger K will result in lower heterogeneity than a single run with smaller K, but heterogeneity will always stay the same of decrease as you increase $k$.)  Let's explore this general trend for ourselves by performing the following analysis.\n",
                "\n",
                "\n",
                "This code block will take some times to complete. It will try 5 values of `k` and for each `k`, will try 3 different seeds. The cell will print its progress to help you know how far it has made. When `k` is larger, it will take longer to run (why might that be?)!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 56,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%time\n",
                "\n",
                "def plot_k_vs_heterogeneity(k_values, heterogeneity_values):\n",
                "    \"\"\"\n",
                "    Given list of k-values and their heterogeneities, will make a plot\n",
                "    showing how heterogeneity varies with k.\n",
                "    \"\"\"\n",
                "    plt.figure(figsize=(7,4))\n",
                "    plt.plot(k_values, heterogeneity_values, linewidth=4)\n",
                "    plt.xlabel('K')\n",
                "    plt.ylabel('Heterogeneity')\n",
                "    plt.title('K vs. Heterogeneity')\n",
                "    plt.rcParams.update({'font.size': 16})\n",
                "    plt.tight_layout()\n",
                "\n",
                "all_centroids = {}\n",
                "all_cluster_assignment = {}\n",
                "heterogeneity_values = []\n",
                "seeds = [20000, 40000, 80000]\n",
                "k_list = [2, 10, 25, 50, 100]\n",
                "\n",
                "for k in k_list:\n",
                "    print(f'Running k = {k}')\n",
                "    heterogeneity = []\n",
                "    all_centroids[k], all_cluster_assignment[k] = kmeans_multiple_runs(tf_idf, k, max_iter=400,\n",
                "                                                                       seeds=seeds, verbose=True)\n",
                "    score = compute_heterogeneity(tf_idf, k, all_centroids[k], all_cluster_assignment[k])\n",
                "    heterogeneity_values.append(score)\n",
                "\n",
                "plot_k_vs_heterogeneity(k_list, heterogeneity_values)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Inspecting clusters of documents\n",
                "Let's start visualizing some clustering results to see if we think the clustering makes sense.  We can use such visualizations to help us assess whether we have set K too large or too small for a given application.  Following the theme of this course, we will judge whether the clustering makes sense in the context of document analysis.\n",
                "\n",
                "What are we looking for in a good clustering of documents?\n",
                "* Documents in the same cluster should be similar.\n",
                "* Documents from different clusters should be less similar.\n",
                "\n",
                "So a bad clustering exhibits either of two symptoms:\n",
                "* Documents in a cluster have mixed content.\n",
                "* Documents with similar content are divided up and put into different clusters.\n",
                "\n",
                "To help visualize the clustering, we do the following:\n",
                "* Fetch nearest neighbors of each centroid from the set of documents assigned to that cluster. We will consider these documents as being representative of the cluster.\n",
                "* Print titles and first sentences of those nearest neighbors.\n",
                "* Print top 5 words that have highest tf-idf weights in each centroid."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 59,
            "metadata": {},
            "outputs": [],
            "source": [
                "def visualize_document_clusters(wiki, tf_idf, centroids, cluster_assignment, k, words, \n",
                "                                display_docs=5):\n",
                "    \"\"\"\n",
                "    Given a set of clustered documents, prints information about the centroids including\n",
                "       - The title and starting sentence of the closest 5 points to each centroid\n",
                "       - The five words that are contained in the clusters documents with the highest TF-IDF.\n",
                "    \n",
                "    Parameters:  \n",
                "      - wiki: original dataframe\n",
                "      - tf_idf: data matrix containing TF-IDF vectors for each document\n",
                "      - centroids: A np.array of length k that contains the centroids for the clustering\n",
                "      - cluster_assignments: A np.array of length N that has the cluster assignments for each row\n",
                "      - k: What value of k is used\n",
                "      - words: List of words in the corpus (should match tf_idf)\n",
                "      - display_odcs: How many documents to show for each cluster (default 5)\n",
                "    \"\"\"\n",
                "    print('=' * 90)\n",
                "\n",
                "    # Visualize each cluster c\n",
                "    for c in range(k):\n",
                "        # Cluster heading\n",
                "        print(f'Cluster {c}  ({(cluster_assignment == c).sum()} docs)'),\n",
                "        # Print top 5 words with largest TF-IDF weights in the cluster\n",
                "        idx = centroids[c].argsort()[::-1]\n",
                "        for i in range(5): # Print each word along with the TF-IDF weight\n",
                "            print(f'{words[idx[i]]}:{centroids[c,idx[i]]:.3f}', end=' '),\n",
                "        print()\n",
                "        \n",
                "        if display_docs \u003e 0:\n",
                "            print()\n",
                "            # Compute distances from the centroid to all data points in the cluster,\n",
                "            # and compute nearest neighbors of the centroids within the cluster.\n",
                "            distances = pairwise_distances(tf_idf, centroids[c].reshape(1, -1), metric='euclidean').flatten()\n",
                "            distances[cluster_assignment!=c] = float('inf') # remove non-members from consideration\n",
                "            nearest_neighbors = distances.argsort()\n",
                "            # For the nearest neighbors, print the title as well as first 180 characters of text.\n",
                "            # Wrap the text at 80-character mark.\n",
                "            for i in range(display_docs):\n",
                "                text = ' '.join(wiki.iloc[nearest_neighbors[i]]['text'].split(None, 25)[0:25])\n",
                "                print(f'* {wiki.iloc[nearest_neighbors[i]][\"name\"]:50s} {distances[nearest_neighbors[i]]:.5f}')\n",
                "                print(f'  {text[:90]}')\n",
                "                if len(text) \u003e 90:\n",
                "                    print(f'  {text[90:180]}')\n",
                "                print()\n",
                "        print('=' * 90)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Let us first look at the 2 cluster case (K=2)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 60,
            "metadata": {},
            "outputs": [],
            "source": [
                "k = 2\n",
                "visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Both clusters have mixed content, although it appears that cluster 0 has women and cluster 1 has men.\n",
                "\n",
                "It would be better if we sub-divided into more categories. So let us use more clusters. How about `K=10`?"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 61,
            "metadata": {
                "scrolled": true
            },
            "outputs": [],
            "source": [
                "k = 10\n",
                "visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k, words)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "We no longer have the clear split between men and women. Cluters 0 and 2 appear to be still mixed, but others are quite consistent in content.\n",
                "* Cluster 0: notable women\n",
                "* Cluster 1: baseball players\n",
                "* Cluster 2: researchers, professors\n",
                "* Cluster 3: football(soccer)\n",
                "* Cluster 4: musicians, singers, song writers\n",
                "* Cluster 5: golfers\n",
                "* Cluster 6: painters, scultpers, artists\n",
                "* Cluster 7: orchestral musicians, conductors\n",
                "* Cluster 8: politicians, political personel\n",
                "* Cluster 9: film directors\n",
                "\n",
                "Clusters are now more pure (only having one category), but some are qualitatively \"bigger\" than others. For instance, the category of scholars is more general than the category of film directors. Increasing the number of clusters may split larger clusters. Another way to look at the size of cluster is to count the number of articles in each cluster."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 62,
            "metadata": {},
            "outputs": [],
            "source": [
                "np.bincount(all_cluster_assignment[10])"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "There appears to be at least some connection between the topical consistency of a cluster and the number of its member data points.\n",
                "\n",
                "Let us visualize the case for K=25. For the sake of brevity, we do not print the content of documents. Let's print out the top words with highest TF-IDF weights, to understand what type of documents tend to be in the clusters.\n",
                ""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 63,
            "metadata": {},
            "outputs": [],
            "source": [
                "k = 25\n",
                "visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k,\n",
                "                            words, display_docs=0) # turn off text for brevity"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Looking at the representative examples and top words, we classify each cluster as follows. Notice the bolded items, which indicate the appearance of a new theme that did not appear to be represented with $k=10$.\n",
                "* Cluster 0: **British labor party**\n",
                "* Cluster 1: **Bishops**\n",
                "* Cluster 2: **danish CEOs**\n",
                "* Cluster 3: baseball\n",
                "* Cluster 4: politicials\n",
                "* Cluster 5: **psychology researchers**\n",
                "* Cluster 6: **medical researchers**\n",
                "* Cluster 7: **republican politicians**\n",
                "* Cluster 8: football(soccer)\n",
                "* Cluster 9: **prime ministers**\n",
                "* Cluster 10: golfers\n",
                "* Cluster 11: coaches\n",
                "* Cluster 12: **lawers**\n",
                "* Cluster 13: researchers, professors\n",
                "* Cluster 14: writers\n",
                "* Cluster 15: artists, museaum workers\n",
                "* Cluster 16: film directors\n",
                "* Cluster 17: musicians\n",
                "* Cluster 18: **airforce commanders**\n",
                "* Cluster 19: orchestral musicians\n",
                "* Cluster 20: *unclear*\n",
                "* Cluster 21: *unclear*\n",
                "* Cluster 22: *unclear*\n",
                "* Cluster 23: politicians\n",
                "* Cluster 24: **hockey players**\n",
                "\n",
                "Indeed, increasing K achieved the desired effect of breaking up large clusters.  Depending on the application, this may or may not be preferable to the K=10 analysis.\n",
                "\n",
                "Let's take it to the extreme and set K=100. We have a suspicion that this value is too large. Let us look at the top words from each cluster:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 64,
            "metadata": {},
            "outputs": [],
            "source": [
                "k = 100\n",
                "visualize_document_clusters(wiki, tf_idf, all_centroids[k], all_cluster_assignment[k], k,\n",
                "                            words, display_docs=0)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**A high value of K encourages pure clusters, but we cannot keep increasing K. For large enough K, related documents end up going to different clusters.**\n",
                "\n",
                "That said, the result for K=100 is not entirely bad. After all, it gives us separate clusters for such categories as Scotland, Brazil, LGBT, computer science and the Mormon Church. If we set K somewhere between 25 and 100, we should be able to avoid breaking up clusters while discovering new ones.\n",
                "\n",
                "Also, we should ask ourselves how much **granularity** we want in our clustering. If we wanted a rough sketch of Wikipedia, we don't want too detailed clusters. On the other hand, having many clusters can be valuable when we are zooming into a certain part of Wikipedia.\n",
                "\n",
                "**There is no golden rule for choosing K. It all depends on the particular application and domain we are in.**\n",
                "\n",
                "Another heuristic people use that does not rely on so much visualization, which can be hard in many applications (including here!) is as follows.  Track heterogeneity versus K and look for the **\"elbow\"** of the curve where the heterogeneity decrease rapidly before this value of K, but then only gradually for larger values of K.  This naturally trades off between trying to minimize heterogeneity, but reduce number of clusters.  In the heterogeneity versus K plot made above, we did not yet really see a flattening out of the heterogeneity, which might indicate that indeed K=100 is \"reasonable\" and we only diminishing returns for larger values of K (which are even harder to visualize using the methods we attempted above.)"
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 🔍 **Question 8** Small clusters\n",
                "\n",
                "Another sign of too large K is having lots of small clusters. Look at the distribution of cluster sizes (by number of member data points). When doing k-means with $k=100$, how many of the clusters have fewer than 44 articles (i.e. 0.004% of the dataset)?\n",
                "\n",
                "Save your result in a variable called `num_small_clusters`\n",
                ""
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 65,
            "metadata": {},
            "outputs": [],
            "source": [
                "### edTest(test_q8_small_clusters) ###\n",
                "\n",
                "# TODO count clusters with fewer than 44 articles\n",
                "num_small_clusters = ..."
            ]
        },
        {
            "attachments": {},
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Keep in mind though that tiny clusters aren't necessarily bad. A tiny cluster of documents that really look like each others is definitely preferable to a medium-sized cluster of documents with mixed content. However, having too few articles in a cluster may lead us to question if that cluster is really worth separating from the others."
            ]
        }
    ]
}
