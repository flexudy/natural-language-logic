# Towards Neuro-Symbolic Language Understanding

![alt text](https://www.flexudy.com/wp-content/uploads/2021/09/conceptor.png "Flexudy's conceptor")

At [Flexudy](https://flexudy.com), we look for ways to unify symbolic and sub-symbolic methods to improve model interpretation and inference.

## Problem

1. Word embeddings are awesome 🚀. However, no one really knows what an array of 768 numbers means?
2. Text/Token classification is also awesome ❤️‍. Still, classifying things into a finite set of concepts is rather limited.
3. Last but not least, how do I know that the word *cat* is a **mammal** and also an **animal** if my neural network is only trained to predict whether something is an animal or not?

## Solution

1. It would be cool if my neural network would just know that **cat** is an **animal** right? *∀x.Cat(x) ⇒ Animal(x)*.
Or for example, (*∀x.SchöneBlumen(x) ⇒ Blumen(x)*) -- English meaning: For all x, If x is a beautiful flower, then x is still a flower. --

2. All of a sudden, tasks like **Question Answering**, **Summarization**, **Named Entity Recognition** or even **Intent Classification** etc become easier right?

Well, one might probably still need time to build a good and robust solution that is not as large as **GPT3**.

Like [Peter Gärdenfors, author of conceptual spaces](https://www.goodreads.com/book/show/1877443.Conceptual_Spaces), we are trying to find ways to navigate between the symbolic and the sub-symbolic by thinking in concepts.

Should such a solution exist, one could easily leverage true logical reasoning engines on natural language.

How awesome would that be? 💡

## Flexudy's Conceptor

1. We developed a poor man's implementation of the ideal solution described above.
2. Though it is a poor man's model, **it is still a useful one** 🤗.

### Usage

No library should anyone suffer. Especially not if it is built on top of 🤗 **HF Transformers**.


Go to the [Github repo](https://github.com/flexudy/natural-language-logic)


```python
from flexudy.conceptor.start import FlexudyConceptInferenceMachineFactory

# Load me only once
concept_inference_machine = FlexudyConceptInferenceMachineFactory.get_concept_inference_machine()

# A list of terms.
terms = ["cat", "dog", "economics and sociology", "public company"]

# If you don't pass the language, a language detector will attempt to predict it for you
# If any error occurs, the language defaults to English.
language = "en"

# Predict concepts
# You can also pass the batch_size=2 and the beam_size=4
concepts = concept_inference_machine.infer_concepts(terms, language=language)
```

Output:

```python
{'cat': ['mammal', 'animal'], 'dog': ['hound', 'animal'], 'economics and sociology': ['both fields of study'], 'public company': ['company']}
```

### How was it trained?

1. Using Google's T5-base and T5-small. Both models are released on the Hugging Face Hub.
2. T5-base was trained for only two epochs while T5-small was trained for 5 epochs.

## Where did you get the data?

1. I extracted and curated a fragment of [Conceptnet](https://conceptnet.io/)
2. In particular, only the IsA relation was used.
3. Note that one term can belong to multiple concepts (which is pretty cool if you think about [Fuzzy Description Logics](https://lat.inf.tu-dresden.de/~stefborg/Talks/QuantLAWorkshop2013.pdf)).
Multiple inheritances however mean some terms belong to so many concepts. Hence, I decided to randomly throw away some due to the **maximum length limitation**.

### Setup
1. I finally allowed only `2` to `4` concepts at random for each term. This means, there is still great potential to make the models generalise better 🚀.
3. I used a total of `279884` training examples and `1260` for testing. Edges -- i.e `IsA(concept u, concept v)` -- in both sets are disjoint.
4. Trained for `15K` steps with learning rate linear decay during each step. Starting at `0.001`
5. Used `RAdam Optimiser` with weight_decay =`0.01` and batch_size =`36`.
6. Source and target max length were both `64`.

### Multilingual Models

1. The "conceptor" model is multilingual. English, German and French is supported.
2. [Conceptnet](https://conceptnet.io/) supports many languages, but I just chose those three because those are the ones I speak.

### Metrics for flexudy-conceptor-t5-base

| Metric        |         Score |
| ------------- |:-------------:|
| Exact Match   | 36.67         |
| F1            | 43.08         |
| Loss smooth   | 1.214         |

Unfortunately, we no longer have the metrics for flexudy-conceptor-t5-small. If I recall correctly, base was just slightly better on the test set (ca. `2%` F1).

## Why not just use the data if you have it structured already?

Conceptnet is very large. Even if you just consider loading a fragment into your RAM, say with only 100K edges, this is still a large graph.
Especially, if you think about how you will save the node embeddings efficiently for querying.
If you prefer this approach, [Milvus](https://github.com/milvus-io/pymilvus) can be of great help.
You can compute query embeddings and try to find the best match. From there (after matching), you can navigate through the graph at `100%` precision.
