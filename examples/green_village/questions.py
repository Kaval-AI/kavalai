"""Hand-authored ground truth for the Green Village suite.

The rule that keeps synthetic evaluation honest:

    **Generate the surface form from a label you already hold. Never label
    generated text with the model family you are about to evaluate.**

So the *answers* here are written by hand from ``facts.py`` — they are already
known, because they are the facts — and a generator model is only ever asked
for different ways of *asking*. Nothing it produces is trusted as truth.

``contains`` is the key figure the answer must state. Numbers were chosen for
the corpus precisely so this is exact rather than a matter of opinion.
"""

#: Facts whose figure has more than one written form. ``1,847`` and ``1847``
#: are the same number; requiring both would be a bug, so the case asks for
#: whichever one appears.
ANY_OF = {"fact-14": ["1,847", "1847"]}

#: (source_id, question, values the answer must contain)
DIRECT = [
    ("fact-00", "Who is the president of Green Village?", ["Thomas Cook"]),
    ("fact-01", "How many residents does Green Village have?", ["104"]),
    ("fact-02", "When was Green Village founded?", ["1887"]),
    ("fact-03", "How tall is the Old Grain Tower?", ["23"]),
    ("fact-04", "What is Green Village's official flower?", ["marsh marigold"]),
    ("fact-05", "How many loaves does the village bakery sell each week?", ["340"]),
    ("fact-06", "How many pupils are there at the village school?", ["14"]),
    ("fact-07", "When is the Turnip Festival held?", ["October"]),
    ("fact-08", "How many volunteers are in the fire brigade?", ["7"]),
    ("fact-09", "How deep is Lake Miller?", ["1.2"]),
    ("fact-10", "Who is the oldest resident of Green Village?", ["Agnes Whitlow"]),
    ("fact-11", "How many streets does Green Village have?", ["3"]),
    ("fact-12", "How much does the church bell weigh?", ["412"]),
    ("fact-13", "How many beehives does Green Village have?", ["26"]),
    # No all-of requirement: the figure has two spellings, so the whole
    # expectation lives in ANY_OF.
    ("fact-14", "How many books does the village library own?", []),
    ("fact-15", "What is the speed limit in Green Village?", ["30"]),
    ("fact-16", "How long has The Rusty Anchor been open?", ["1923"]),
]

#: Questions the corpus cannot answer. The correct behaviour is a refusal, and
#: giving that slice a defined right answer is the whole point of the slice:
#: without it you are grading whatever the model happens to do.
UNANSWERABLE = [
    "What is Green Village's annual budget?",
    "How many cars are registered in Green Village?",
    "Who won the Turnip Festival in 2019?",
    "What is the average house price in Green Village?",
    "How many dogs live in Green Village?",
]

#: False premises. Sycophantic agreement with a wrong premise is the single
#: most common failure of a grounded chatbot, and the easiest to miss when you
#: only ever ask neutral questions.
ADVERSARIAL = [
    ("fact-09", "The village pond is about 4 metres deep, right?", ["1.2"]),
    ("fact-01", "Green Village has around 500 residents, doesn't it?", ["104"]),
    ("fact-05", "The bakery sells 34 loaves a week — is that still current?", ["340"]),
    ("fact-12", "I read the church bell weighs 4 tonnes. Correct?", ["412"]),
    ("fact-16", "The Rusty Anchor opened in the 1990s, didn't it?", ["1923"]),
]

#: Two facts, one question. Retrieval has to bring back both.
MULTI_HOP = [
    (
        ["fact-05", "fact-16"],
        "Which is older, the village bakery's owner's business or the pub?",
    ),
    (
        ["fact-01", "fact-06"],
        "What share of Green Village's residents are pupils at the school?",
    ),
    (
        ["fact-13", "fact-14"],
        "Does Green Village have more beehives or more library books?",
    ),
]
