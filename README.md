
## Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?

### Zorik Gekhman ### T ### ∗ ### Gal Yona ### G ### Roee Aharoni ### G ### Matan Eyal ### G ### Amir Feder ### G
 Roi Reichart T Jonathan Herzig G

_T_ Technion - Israel Institute of Technology _G_ Google Research
### zorikgekhman@gmail.com, jherzig@google.com

 Abstract


When large language models are aligned via
supervised fine-tuning, they may encounter
new factual information that was not acquired
through pre-training. It is often conjectured
that this can teach the model the behavior of
_hallucinating_ factually incorrect responses, as
the model is trained to generate facts that are
not grounded in its pre-existing knowledge.
In this work, we study the impact of such ex
posure to new knowledge on the capability of
the fine-tuned model to utilize its pre-existing
knowledge. To this end, we design a controlled
setup, focused on closed-book QA, where we
vary the proportion of the fine-tuning examples
that introduce new knowledge. We demonstrate
that large language models struggle to acquire
new factual knowledge through fine-tuning, as
fine-tuning examples that introduce new knowledge are learned significantly slower than those
consistent with the model’s knowledge. How
ever, we also find that as the examples with
new knowledge are eventually learned, they
linearly increase the model’s tendency to hal
lucinate. Taken together, our results highlight
the risk in introducing new factual knowledge
through fine-tuning, and support the view that
large language models mostly acquire factual
knowledge through pre-training, whereas fine
tuning teaches them to use it more efficiently.

### 1 ### Introduction

Pre-training Large Language Models (LLMs) on
textual corpora embeds substantial factual knowl
edge in their parameters ( Petroni et al. , 2019 ;
AlKhamissi et al. , 2022 ; Cohen et al. , 2023 ), which
is essential for excelling in various downstream
applications. These models often require further
alignment to desired behaviors, typically achieved
through supervised fine-tuning on instructionfollowing tasks ( Wei et al. , 2022 ; Mishra et al. ,
2022 ) and preference learning from human feed
back ( Ouyang et al. , 2022 ; Rafailov et al. , 2024 ).

_∗_ Work done during an internship at Google Research.


![2405.05904v1.pdf-0-0.png](2405.05904v1.pdf-0-0.png)

Figure 1: Train and development accuracies as a func
tion of the fine-tuning duration, when fine-tuning on
50% `Known` and 50% `Unknown` examples. `Unknown` ex
amples are fitted substantially slower than `Known` . The
best development performance is obtained when the
LLM fits the majority of the `Known` training examples
but only few of the `Unknown` ones. From this point,
fitting `Unknown` examples reduces the performance.

In the fine-tuning phase, the model is usually
trained on outputs created by human annotators
or other LLMs. As a result, the model may en
counter new factual information, extending beyond
the knowledge it acquired during pre-training. This
raises the question of how LLMs integrate new
facts outside of their pre-existing knowledge. One
possibility is that the model simply adapts by learn
ing this new factual information. However, a com
mon conjecture posits that such exposure to new
knowledge may encourage the model to _hallucinate_ factually incorrect responses, as the model
is essentially trained to generate facts that are not
grounded in its pre-existing knowledge ( Schulman ,
2023 ; Huang et al. , 2023 ; Gao , 2021 ; Goldberg ,
2023 ; Gudibande et al. , 2023 ).
In this work, we study how learning new factual


-----


knowledge through fine-tuning impacts the model’s
tendency to hallucinate w.r.t. its pre-existing knowl
edge, exploring the above conjecture. 1

To study the impact of new knowledge, we must
be able to assess whether a single fine-tuning example is consistent with the model’s knowledge.
We propose _knowledge categories_ **S** li , derived from a continuous mea **CK** , a hierarchy of four
sure that quantifies the agreement between model
generated answers and the ground-truth labels. In
**S** li **CK** , examples are first categorized into `Known`
and `Unknown` types, where the latter corresponds to
examples with facts that are most likely unknown
to the model. The `Known` examples are subse
quently split into three categories: `HighlyKnown` ,
`MaybeKnown` , and `WeaklyKnown` (Figure 2 ).
Equipped with the above method, we carefully
design a controlled study, focused on closed-book
question answering (QA), where we vary the pro
portion of the fine-tuning examples categorized as
`Unknown` , while controlling for other factors.
Our study empirically demonstrates that learn
ing from `Unknown` fine-tuning examples is linearly
correlated with the model’s tendency to _hallucinate_
w.r.t. its pre-existing knowledge (§ 4 ). Conversely,
learning from `Known` examples is correlated with
better utilization of pre-existing knowledge.
Through an analysis of the training dynamics,
we discover that the LLM fits `Unknown` fine-tuning
examples _substantially slower_ than `Known` exam
ples (top plot in Figure 1 ). This indicates that dur
ing fine-tuning, LLMs struggle to integrate new
factual knowledge (present in the `Unknown` fine
tuning examples). Instead, they mostly learn to ex
pose their pre-existing knowledge (using the `Known`
fine-tuning examples).
From a practical perspective, mitigating overfitting using _early-stopping_ (vertical dotted line
in Figure 1
) can minimize the risk of the hallucinations caused by fitting the `Unknown` examples,
since they primarily emerge in in later training
stages as a form of overfitting (as illustrated by
the development performance decline in the bot
tom plot of Figure 1 ). Alternatively, we also show
that fine-tuning examples substantially reduces the risk of overfitting, _filtering-out_ the `Unknown`
without sacrificing performance.
We further evaluate the impact of fine-tuning
examples from each of our three `Known` knowl-

1 While we focus on supervised fine-tuning, our findings
are relevant to offline preference optimization methods such
as DPO ( Rafailov et al. , 2024 ) that may add new knowledge.


edge categories on performance (§ 5 ). Unexpect
edly, we find that a model fine-tuned only on exam
ples from the highest knowledge degree, denoted
`HighlyKnown` , does not yield the best results. Our
analysis reveals that incorporating `MaybeKnown`
fine-tuning examples, representing facts with lower
degrees of certainty, plays an important part in properly handling such examples in test time. This indi
cates that the composition of fine-tuning examples
significantly influences the extent to which LLMs
effectively utilize their pre-existing knowledge.
To summarize, we study the effect of new factual
knowledge in the fine-tuning data by designing a
controlled setup that isolates this factor. We find
that fine-tuning examples that introduce new knowledge are learned slowly, which suggests that LLMs
struggle to integrate new knowledge through fine
tuning and supports the view that LLMs mostly ac
quire knowledge through pre-training ( Zhou et al. ,
2023 ; Lin et al. , 2023 ). However, we also find
that as the model eventually learns new knowledge
through fine-tuning, it becomes more prone to hal
lucinations w.r.t. its pre-existing knowledge. Col
lectively, our findings highlight the potential for
unintended consequences when introducing new
knowledge through fine-tuning, and imply that fine
tuning may be more useful as a mechanism to en
hance the utilization of pre-existing knowledge.

### 2 ### Study Setup

Given a fine-tuning dataset _D_ and a pre-trained
LLM _M_ , we denote by _M_ _D_ a model obtained by
fine-tuning _M_ on _D_ . To study how new knowledge
in _D_ affects _M_ _D_ ’s performance, we design a con
trolled setup creating variants of _D_ with varying
proportions of examples that are unknown to _M_ .
When constructing _D_ , our objective is to reflect
instruction tuning on diverse knowledge-intensive
tasks while maintaining control over the experimen
tal setting. We thus focus on factual knowledge
that can be structured as _(subject, relation, object)_
triplets, which are converted into closed-book QA
format. In this setup, _D_ = ( _q_ _i_ _, a_ _i_ ) _N_ _i_ =1 , where _q_
_{_ _}_
is a knowledge-seeking question corresponding to
a specific triplet (e.g., “ _Where is Paris located?_ ”)
and _a_ is the ground-truth answer (e.g., “ _France_ ”).
To this end, we use E NTITY Q UESTIONS ( Sciavolino
et al. , 2021 ), where triplets from a diverse set of
relations from Wikidata ( Vrandeˇ ci´ c and Krötzsch ,

2014 ) are converted to QA pairs. These relations
encompass a broad spectrum of factual knowledge,


-----


|Type|Category|Definition|Explanation|
|---|---|---|---|
|Known|HighlyKnown|PCorrect(q, a; M, T = 0) = 1|Greedy decoding always predicts the correct answer.|
||MaybeKnown|PCorrect(q, a; M, T = 0) ∈(0, 1)|Greedy decoding sometimes (but not always) predicts the correct answer.|
||WeaklyKnown|PCorrect(q, a; M, T = 0) = 0 ∧ PCorrect(q, a; M, T > 0) > 0|Greedy decoding never predicts the correct answer, whereas temperature sampling with T > 0 sometimes predicts the correct answer.|
|Unknown|Unknown|PCorrect(q, a; M, T ≥0) = 0|The model never predicts the correct answer, thus it seem to lack the knowledge of the correct answer.|


|Ernest Holmes|, ..|Ernest Holmes|
|---|---|---|


|Col1|Col2|(a)|Col4|Col5|Col6|
|---|---|---|---|---|---|
|Category|Question|Gold Answer||Greedy Answers|Sampled Answers|
|HighlyKnown MaybeKnown WeaklyKnown Unknown|Who founded Science of Mind? What is the capital of Toledo District? What kind of work does Scott McGrew do? Where is Benedict located?||Ernest Holmes|[Ernest Holmes, .. Ernest Holmes, ..] [Belmopan, .., Punta Gorda, ..] [Film director, .. Actor, ..] [Louisiana, .. New Mexico, ..]|[..., ...] [..., ...] [Musician, .. Journalist, ..] [Washington, .. Texas, ..]|
||||Punta Gorda|||
||||Journalist|||
||||Hubbard County|||


(b)

Figure 2: Formal definitions of the **S** li **CK** knowledge categories, based on the _P_ `Correct` measure as defined in § 3 **(a)** ,
accompanied with real examples from the annotated E NTITY Q UESTIONS dataset used in our study **(b)** .


including biographical information, geographical
data, ownership and authorship details, history and
more. We use the original development and test
splits, and we sub-sample the train split to create
different variants of _D_ . We focus on 12 diverse
relations and reserve 7 additional relations for an
_out-of-distribution_ test set, used (only) in § 4.5 .
As _M_ , we use the PaLM 2-M base model ( Anil
et al. , 2023 ). We focus on exact match (EM) as our
evaluation metric. 2 Full technical details are in § A .

### 3 ### Quantifying Knowledge in LLMs

To assess the effect of new knowledge in _D_ on
the performance of _M_ _D_ , we have to annotate each
( _q, a_ ) pair in _D_ w.r.t. whether _M_ knows that the
answer to _q_ is _a_ . To estimate this, we define a con
tinuous _P_ `Correct` measure based on samples from
_M_ , and use it to divide ( _q, a_ ) pairs into four _knowl_
_edge categories_ . We name this approach **S** li **CK**
( **S** ampling-based **C** ategorization of **K** nowledge).

**Defining** **_P_** `Correct` **.** We adopt the perspective that
_M_ _knows_ the answer to _q_ is _a_ if it generates _a_ when
prompted to answer _q_ ( Kadavath et al. , 2022 ; Man
akul et al. , 2023 ). Since _M_ is a base model that
has not been specifically fine-tuned to follow in
structions, we prompt _M_ using in-context learning
with few-shot exemplars. Following Rubin et al.
( 2022 ), we make sure that the few-shot exemplars

3
have high semantic similarity to _q_ .

In practice, _M_ can predict different answers
since (1) the choice of exemplars influences in-

2 We validated that in our setting EM strongly correlates
with word-level F1 ( Rajpurkar et al. , 2016 ), and we choose
EM as it is more intuitive for the purposes of our analysis.
3 In our study we achieve this by using exemplars from
the same relation. E.g., if _q_ = “ _Where is Paris located?_ ”, the
exemplars would follow the pattern “ _Where is {X} located?_ ”.


dividual predictions and (2) temperature sampling,
if used, introduces randomness. To reflect this, we
define _P_ `Correct` ( _q, a_ ; _M, T_ ) as an estimate of how
likely is _M_ to accurately generate the correct an
swer _a_ to _q_ , when prompted with _random few-shot_
_exemplars_ and using decoding temperature _T_ .
For the purposes of our study we approximate the value of _P_ `Correct` using _N_ ex = 10
different random 4-shot prompts. 4 For each
4-shot prompt, we predict the greedy answer
using _T_ = 0 and 16 sampled answers using
_T_ = 0 _._ 5 . _P_ `Correct` ( _q, a_ ; _M, T_ = 0) is estimated
by the fraction of correct greedy answers, and
_P_ `Correct` ( _q, a_ ; _M, T >_ 0) by the fraction of cor
rect sampled answers. Full details are in § C .

**Deriving knowledge categories from** **_P_** `Correct` **.**
We define the `Unknown` category (bottom row
in Figures 2a and 2b ) to represent ( _q, a_ ) pairs
for which _M_ _never_
predicts the correct answer to _q_ . In our notations this means that
_P_ `Correct` ( _q, a_ ; _M, T_ 0) = 0 . Alternatively, if
_≥_
_P_ `Correct` ( _q, a_ ; _M, T_ 0) _>_ 0 , i.e. _M_ _sometimes_
_≥_
predicts the correct answer to _q_ , we consider ( _q, a_ )
as `Known` . In this choice, we posit that if prompting
_M_ to answer _q_ can _sometimes_ result with the cor
rect answer _a_ , then _M_ must have some association
with the relevant fact.
Recognizing that knowledge can vary in degrees
of certainty and extent, we divide the `Known` ( _q, a_ )
pairs into three distinct categories (top three rows
in Tables 2a and 2b ). Motivated by the principle
that _M_ should _consistently_ predict _a_ if ( _q, a_ ) is
`Known` , we put emphasis on _greedy decoding_ out
comes, represented with _P_ `Correct` ( _q, a_ ; _M, T_ = 0) .

4 We use 4-shot simply since we found it enough for _M_ to
output answers in the correct format.


-----


(a) (b)

![2405.05904v1.pdf-3-0.png](2405.05904v1.pdf-3-0.png)

![2405.05904v1.pdf-3-1.png](2405.05904v1.pdf-3-1.png)

Figure 3: Test performance as a function of the % of `Unknown` examples in the fine-tuning dataset _D_ . In **(a)** ,
each line corresponds to a different (fixed) number of epochs, except the EARLY _ STOP , which corresponds to
early-stopping using the development set (see § 4 ). In **(b)** we present the ablation from § 4.2 . Full lines correspond
to fine-tuning on _D_ and are identical to (a). Dotted lines correspond to fine-tuning on the ablated variants _D_ `Known` ,
where `Unknown` examples are filtered-out. For 0% `Unknown` _D_ = _D_ `Known` and for 100% `Unknown` there is no _D_ `Known` .


`HighlyKnown` represents ( _q, a_ ) pairs for which _M_
_always_ greedily predicts _a_ . If _M_ _sometimes_ (but
not always) greedily predicts _a_ , we consider ( _q, a_ )
as `MaybeKnown` . Lastly, if _M_ _never_ greedily pre
dicts _a_ , we classify ( _q, a_ ) as `WeaklyKnown` .
We apply **S** li **CK** to annotate each ( _q, a_ ) pair in

5
our dataset with its knowledge category w.r.t. _M_ .

We analyze the quality of our categories in § 6 .

### 4 ### How Harmful are `Unknown` ### Examples?

In this section we study the effect of new knowl
edge in the fine-tuning dataset _D_ on performance.
To isolate this effect, we vary the proportion of
`Unknown` examples in _D_ , while controlling for
other factors. Specifically, we fix _|_ _D_ _|_ and create
variants of _D_ with _X_ % of `Unknown` and (100 _−_
_X_ )% `Known` examples (full details in § E ). We treat
the `Known` categories collectively (see Figure 2a ),
and provide a per-category analysis in § 5 . We de
note early-stopping based on the development set
as EARLY _ STOP (happens after 5-10 epochs) and 50
fine-tuning epochs as C ONVERGENCE , as at this point
_M_ always completely fits _D_ (i.e. 100% training
accuracy). We measure test performance as a proxy
for hallucinations since we are in a closed-book QA
setup with disjoint train/test splits, where the model
has to use its per-existing knowledge to answer test
questions (see § B for further discussion).

5 In E NTITY Q UESTIONS we have 24% `HighlyKnown` ,
23% `MaybeKnown` , 17% , `WeaklyKnown` , and 36% `Unknown` .
Full per-relation statistics are in § D .


**4.1** **Higher** `Unknown` **Ratio is Proportional to**
**Performance Degradation**

Figure 3a presents the performance as a function
of the % of `Unknown` examples in _D_ , for different
fine-tuning durations. Higher % `Unknown` leads to
performance degradation, regardless of the finetuning duration, which indicates that `Unknown`
examples are less useful than `Known` . Perfor
mance is also strongly affected by the fine-tuning
duration, with EARLY _ STOP typically yielding the
best performance. Training for more epochs usually reduces performance (with the lowest perfor
mance observed for C ONVERGENCE ), which can be
attributed to overfitting _D_ . Interestingly, this ef
fect increases with larger % `Unknown` (the inter-line
spacing from EARLY _ STOP exhibits a monotonic in
crease along the positive x-axis), suggesting that a
higher % `Unknown` increases the risk of overfitting.

**4.2** `Unknown` **Examples: Harmful or Neutral?**

Since _|_ _D_ _|_ is fixed, performance drops for higher
% `Unknown` could stem from simply the lower num
ber of the `Known` fine-tuning examples. Thus, it is
still not clear if `Unknown` examples are _harmful_ or
_neutral_ . To address this, we measure the effect of
filtering-out all the `Unknown` examples from _D_ . For
each _D_ variant, we create a corresponding ablated
variant _D_ `Known` , consisting only from the `Known` ex
amples in _D_ . E.g., if _D_ has 25% `Unknown` , we
filter them out and are left with the remaining 75%
`Known` examples and get _D_ `Known` = 0 _._ 75 _D_ .
_|_ _|_ _× |_ _|_
Figure 3b presents the results. Perhaps surpris
ingly, for EARLY _ STOP the results for _D_ are almost


-----


_β_ 0 _β_ kn _β_ unk _R_ 2

In-distribution (§ 4.4 ) 36 _._ 9 7 _._ 3 _−_ 8 _._ 3 0 _._ 86
Out-of-distribution (§ 4.5 ) 36 _._ 2 3 _._ 2 _−_ 3 _._ 0 0 _._ 95

Table 1: Results of our linear model for predicting the
test accuracy as defined by Equation ( 1 ).

as at this point _M_ still did not fit most of them.
Lastly, since `Unknown` examples are the ones that
are likely to introduce new factual knowledge, their
significantly slow fitting rate suggests that LLMs
struggle to acquire new factual knowledge through
fine-tuning, instead they learn to expose their pre
existing knowledge using the `Known` examples.

**4.4** **The Influence of** `Unknown` **vs** `Known` **on**
**Accuracy: A Linear Model Perspective**

Figure 1 demonstrates that after the development
performance peaks at EARLY _ STOP
(vertical dotted line), it deteriorates as _M_ gradually fits more
`Unknown` examples. In this section, we aim to characterize this relationship more accurately by assess
ing whether a simple linear dependency can tie the
impact of fitting training examples on test accuracy. To this end we use the `Known` and `Unknown`
following linear regression model:


![2405.05904v1.pdf-4-0.png](2405.05904v1.pdf-4-0.png)

Figure 4: The state of the examples in the fine-tuning
dataset _D_ after EARLY _ STOP . For each variant of _D_ (y
axis), we illustrate which portion of the examples in _D_
the model fits (i.e. predicts the correct answer for _q_ ).

identical to _D_ `Known` , indicating that the `Unknown`
examples had a _neutral_ effect on performance (as
their removal had minimal impact). Conversely, the
C ONVERGENCE results show that with longer train
ing, `Unknown` examples are actually very _harmful_ .
In this case _D_ under-performs _D_ `Known` , and the gap
between them is proportional to the `Unknown` ratio.
Interestingly, for _D_ `Known` , the gap between

EARLY _ STOP and C ONVERGENCE
is very small (dotted lines), while this gap is very large for _D_ (full
lines). This indicates that the presence of `Unknown`
examples is what makes the variants with higher
`Unknown` ratios more prone to overfitting.

**4.3** `Unknown` **Examples are Fitted Slower than**
`Known` **Examples**

We showed that `Unknown` examples are harmful,
but their negative effect is mostly materialized in
later training stages, and thus can be empirically
avoided using early stopping. To better understand
these trends, we analyze the training dynamics by
examining which fine-tuning examples in _D_ were
fitted by _M_ during various fine-tuning stages. Fig
ure 1 presents the train accuracy of the `Known` and
`Unknown` subsets of _D_ as a function of the fine
tuning duration. The development accuracy is pre
sented in a zoomed-in plot at the bottom, as it falls
within a narrower range. We include a breakdown
of the train accuracy per `Known` category in § F .
_M_ fits `Unknown`
fine-tuning examples substantially slower than `Known` . In EARLY _ STOP (vertical
dotted line), _M_ reaches peak performance on the
development set, while fitting the majority of the
`Known` examples but only a small fraction of the
`Unknown` . In Figure 4 , we show that this behav
ior is consistent across all our variants of _D_ . This
can explain why in EARLY _ STOP the `Unknown` ex
amples had a _neutral_ effect on performance (§ 4.2 ),



_N_ kn
_Accuracy_ = _β_ 0 + _β_ kn
_·_ _D_



_N_ kn unk

+ _β_ unk _N_
_D_ _·_ _D_
_|_ _|_ _|_ _|_


(1)
_|_ _D_ _|_


where _N_ Kn and _N_ Unk are the number of the `Known`
and `Unknown` examples in _D_ that _M_ fits.
We estimate the coefficients 6 by collecting
( _Accuracy_ , _N_ Kn , _N_ Unk ) values after each epoch
from models fine-tuned on all _D_ variants. Table 1
presents the results (top row). The high _R_ 2 indi
cates a strong linear relationship between test accuracy and the type of training examples that are fitted.
Our model entails that fitting `Unknown` examples
hurts performance ( _β_ _unk_ _<_ 0 ), while fitting `Known`
examples improves it ( _β_ kn _>_ 0 ). The estimated
negative impact from `Unknown` roughly matches
the positive impact from `Known` ( _|_ _β_ ukn _| ≈|_ _β_ kn _|_ ).

**4.5** **Generalization to New Relations**

In the above setup, the ( _q, a_ ) pairs in the test set
correspond to triplets with the same set of 12 rela
tions appearing in _D_ . We now investigate whether
our observed dynamics has a broader effect on the
model’s knowledge, and transfers to relations not

6 Full details in § G . We note that this linear model is only
valid in bounded region of _N_ kn _≤|_ _D_ _|_ , _N_ unk _≤|_ _D_ _|_ .


-----


EARLY _ STOP C ONVERGENCE
```
Full Hkn Mkn Wkn Unk Full Hkn Mkn Wkn Unk

```

_D_ `HighlyKnown` 40.5 **98.7** 60.1 9.0 0.6 40.0 **98.4** 58.8 8.5 0.7

_D_ `MaybeKnown` **43.6** **98.4** **69.9** 12.1 1.0 **43.2** 97.5 **68.2** 12.9 1.3

_D_ `WeaklyKnown` 39.2 95.0 59.2 8.6 0.4 35.4 73.5 55.8 **17.2** 2.2


_D_ `Unknown` 37.5 95.6 52.9 6.5 0.6 25.8 55.8 36.6 12.2 **3.2**

_D_ `Natural` **43.5** 98.0 67.6 **14.1** **1.8** 41.8 95.5 61.7 14.8 2.5

Table 2: Accuracies for the single-category variants from § 5 , across per-category subsets of the test set. `Full`
is the original test set (all the categories together). `Hkn` = `HighlyKnown` , `Mkn` = `MaybeKnown` , `Wkn` = `WeaklyKnown` ,
`Unk` = `Unknown` . In each column, the best result is in **bold** , as well as the results for which the difference from the
best is not statistically significant with _p <_ 0 _._ 05 (significance test details are in § I ).


represented in _D_ . To test this, we reserve a subset
of the relations for an _out-of-distribution_ (OOD)
test set, excluding them from the train and develop
ment splits. See § A for details and Tables 4 and 5
for in-distribution vs OOD relations.
Our results on the OOD test set reveal similar key insights: (1) Higher `Unknown` ratio leads
to lower OOD test performance and (2) `Unknown`
examples are harmful for OOD performance, but
mostly when _M_ fits them. A linear model of the
OOD test accuracy (Equation ( 1 )), shows similar
trends: _R_ 2 = 0 _β_ _._ 95 unk (see Table _<_ 0 , _β_ kn 1 _>_ ). More details are in § 0 , _|_ _β_ ukn _| ≈|_ _β_ kn _|_ and H .
Overall, _our insights transfer across relations_ .
This essentially shows that fine-tuning on `Unknown`
examples such as _"Where is [E1] located?"_ , can
encourage hallucinations on seemingly unrelated
questions, such as _"Who founded [E2]?"_ . This
further supports the conclusion that the observed
effects likely stem from the model learning the _be_
_havior_ of generating answers that are not grounded
in its pre-existing knowledge.

### 5 ### Understanding Knowledge Types: Their Value and Impact

When addressing our main research question on
the effect of `Unknown` fine-tuning examples, we
treated the `Known` categories collectively for sim
plicity (see Figure 2a ). We now examine the effect
of each category, exploring the following questions:
**Q1:** from each category impact the test performance? How _training examples_ **Q2:** What is the model’s
performance across _test examples_ from each cate
gory? To address **Q1** we created single-category
variants of the fine-tuning dataset _D_ . A variant of
_D_ consisting solely of examples from the category
`CAT` is denoted as _D_ `CAT` . For reference, we include


a variant with the _natural_ categories distribution in
E NTITY Q UESTIONS , denoted _D_ `Natural` . _D_ is fixed
_|_ _|_
and identical to our experiments in § 4 . To address
**Q2** , we further break down the test set performance
by category. Table 2 presents the results.

**MaybeKnown** **Examples are Essential.** Since
`Unknown` examples are harmful, one might expect
that it would be best to fine-tune on the most exemplary `HighlyKnown` examples. Surprisingly,

_D_ `HighlyKnown`
does not obtain the best overall results, as it excels on `HighlyKnown` test examples,
yet its performance on the remaining categories is
inferior. _D_ `MaybeKnown` yields the best overall perfor
mance. Compared to _D_ `HighlyKnown` , _D_ `MaybeKnown`
enhances _M_ _D_ ’s performance on `MaybeKnown`
( 60 _._ 1 _→_ 69 _._ 9 ), without compromising performance
on `HighlyKnown` ( 98 _._ 7 _→_ 98 _._ 4 ). This suggests
that `MaybeKnown` fine-tuning examples are essen
tial for _M_ _D_ to correctly handle such examples dur
ing inference. It also demonstrates that with the
right fine-tuning examples, _M_ _D_ becomes more ca
pable of utilizing its pre-existing knowledge.

**Limited Knowledge Enhances Overfitting.** In
§ 4.2 , we demonstrated that `Unknown` fine-tuning
examples increase the risk of overfitting. We now
observe that this also applies to `WeaklyKnown` ,
though to a lesser degree. Specifically, at

experience significant performance drops compared to C ONVERGENCE , _D_ `WeaklyKnown` and _D_ `Unknown`

EARLY _ STOP ( 39 _._ 2 _→_ 35 _._ 4 and 37 _._ 5 _→_ 25 _._ 8 ). With
longer training, these variants show a modest improvement on `WeaklyKnown` and `Unknown` , how
ever, they substantially degrade on `HighlyKnown`
and This highlights that the decrease in performance is strongly attributed to an `MaybeKnown` .
increased rate of hallucinations w.r.t. facts that
were already known to _M_ after pre-training.


-----


Interestingly, _D_ `Natural` performs on-par with

_D_ `MaybeKnown` in EARLY _ STOP , suggesting that the
mere presence of `MaybeKnown` examples in _D_ suf
fices for high performance on `MaybeKnown` , even
if _D_ has additional examples from other cate
gories. Yet, _D_ `Natural` ’s performance degrades sig
7
nificantly after C ONVERGENCE , under-performing
_D_ `MaybeKnown` – indicating that it still suffers from
overfitting, most-likely due to the presence of
`WeaklyKnown` and `Unknown`
examples. Taken together these results demonstrate that _D_ `MaybeKnown`
stands out both in terms of top performance and
reduced risk to overfitting.

### 6 ### SliCK Knowledge Categories Analysis

Assessing a model’s knowledge remains an open
problem, particularly since evaluating the quality
of such methods is challenging due to the lack of
ground truth about what the model truly knows. In
this work we proposed **S** li **CK** (§ 3 ): a four-category
classification of facts w.r.t. the model’s knowledge.
We now further analyze and discuss our design
choices, hoping that **S** li **CK** can serve as a useful
taxonomy to guide future research on this subject.

**Fine-grained Known Categories** We first reflect
on whether our choice of splitting `Known` into more
fine-grained categories, based on the greedy de
coding outcome, has been proven meaningful. As
shown in Table 2 , `HighlyKnown` indeed captures
facts with high degree of knowledge, as it consistently exceeds 95% accuracy post fine-tuning,
while `MaybeKnown` and `WeaklyKnown` seem to rep
resent weaker knowledge degrees. As intended,
the performance on `WeaklyKnown` is worse that on
`MaybeKnown` but better than on `Unknown` . Addi
tionally, the _exact_ categories distinction we made
was proven useful since it revealed important in
sights on the importance of the `MaybeKnown` fine
tuning examples, as discussed in detail in § 5 .

**Benchmarking Unknown Test Examples** A de
sired property for ( _q, a_ ) pairs classified as `Unknown`
that appear in the test set, is that _M_ will incorrectly
answer _q_ post fine-tuning (otherwise they are not

8
truly `Unknown` ). In Table 2 we can see that the
accuracy on `Unknown` is extremely low ( 3 _._ 2% or
less), which is a strong indicator that most of the
`Unknown` examples are actually unknown to _M_ .

7 See § I for details about this statistic significance test.
8 Since in our closed-book QA setup the train and test
sets are disjoint, the model has to rely on its pre-existing
knowledge to answer test questions.


![2405.05904v1.pdf-6-0.png](2405.05904v1.pdf-6-0.png)

Figure 5: **S** li **CK** `Unknown` categorization vs. classifying
examples with P(True) _< T_ as `Unknown` . The x-axis
is the % of test examples classified as `Unknown` and
the y-axis is the accuracy on these examples post finetuning. The **yellow line** is P(True) for _T_ _∈_ [0 _,_ 1] . Our
`Unknown` category is the **blue circle** and the **blue line**
corresponds to approximating _P_ `Correct` with less than
10 random 4-shot exemplars (see § 3 and § C ).

As a case study for comparison, we analyze the
P(True) approach by Kadavath et al. ( 2022 ): a con
tinuous score that estimates the probability a model
assigns to the correctness of a specific answer.
P(True) was originally used for _self-evaluating_
model-generated answers, while we use it to as
sess whether _M_ considers the ground-truth answer
as correct. In Figure 5 , we explore classifying ex
amples below a P(True) threshold as `Unknown` and
compare this methodology to **S** li **CK** . Our results in
dicate that, at least in our setting, our approach categorizes `Unknown` examples for which the model’s
performance after fine-tuning is significantly worse.
Specifically, looking at fixed values on the x-axis
shows that if we would label a similar fraction of
test examples as `Unknown` using both methods, the
accuracy on the P(True)-based `Unknown` examples
would be much higher post fine-tuning. 9 Lastly,
the **blue line** shows that using samples from mul
tiple few-shot prompts to approximate _P_ `Correct` is
crucial, as using _N_ ex _<_ 10 leads to higher test
accuracy on **S** li **CK** `Unknown` examples.

### 7 ### Discussion

**Practical Implications.** This work highlights
the risk in using supervised fine-tuning to update
LLMs’ knowledge, as we present empirical evi
dence that acquiring new knowledge through fine
tuning is correlated with hallucinations w.r.t pre
existing knowledge. Additionally, this work raises
important questions for future exploration, regard-

9 This is a preliminary analysis, and we leave a comprehen
sive comparison for future work. More details in § J .


-----


ing fine-tuning practices. We saw that `Unknown` ex
amples are fitted slower than the `Known` ones, thus
their negative effect manifests as a form of _over_
_fitting_ , which emphasizes the importance of using
_early-stopping_ instead of a fixed number of fine
tuning steps. However, early-stopping may be less
effective when fine-tuning on numerous tasks with
distinct optimal stopping points. An alternative so
lution can be to align the fine-tuning data with the
model’s knowledge by filtering-out `Unknown` exam
ples. We show initial evidence that this can reduce
the risk of overfitting without compromising per
formance. A possible drawback of filtering is that
`Unknown` fine-tuning examples can still be useful
to teach LLMs to express uncertainty on `Unknown`
test examples ( Zhang et al. , 2023 ). This raises the
question: _will_ `Unknown` _fine-tuning examples still_
_be harmful if we re-label them with uncertainty ex_
_pressions_ (e.g., _“I don’t know”_ )? Our preliminary
experiment (described in § K ) suggests that the an
swer is _no_ , which indicates that such approaches
could be the most promising. Exploring this is an
interesting direction for future work.

**Superficial Alignment Hypothesis.** Zhou et al.

( 2023 ) hypothesized that the knowledge and ca
pabilities of LLMs are mostly learned during pre
training, while alignment is a simple process where
the model learns the style or format for interacting
with users. They substantiate this hypothesis by
showing that fine-tuning on just `1k` high-quality
examples can result with a competitive assistant
LLM, named LIMA. As discussed in § 4.3 , we
show evidence that LLMs struggle to acquire new
knowledge present in the `Unknown` examples and
mostly learn to utilize their pre-existing knowledge.
We also showed that fine-tuning on `HighlyKnown`
examples led to sub-optimal utilization of preexisting knowledge, despite our task format be
ing simpler than LIMA’s and our dataset being six
times larger. Taken together, our findings suggest
that even though most of the LLM’s knowledge
is indeed acquired through pre-training, the model
learns more than just style or format through fine
tuning, as the selection of fine-tuning examples
significantly influences the model’s capability to
utilize its pre-existing knowledge post fine-tuning.

### 8 ### Related Work

**New knowledge and hallucinations.** Schulman

( 2023 ), Goldberg ( 2023 ) and Gudibande et al.
( 2023 ) mention the conjecture that fine-tuning on


new factual knowledge may encourage hallucina
tions. Huang et al. ( 2023 ) categorized hallucination
causes and formally defined this scenario as _capa_
_bility misalignment_ . They highlight that limited
research addresses capability misalignment due to
the challenge of defining the knowledge boundary
of LLMs. Kang et al. ( 2024 ) showed that when a
fine-tuned LLM encounters unknown queries at test
time, its responses mimic the responses associated
with the unknown examples in the fine-tuning data.
Yin et al. ( 2023 ) showed that LLMs’ performance
is not satisfactory when they face new knowledge
in their input contexts and Lee et al. ( 2023
) analyzed the impact of unknown _in-context_ learning
examples. To the best of our knowledge, our work
is the first to empirically assess the impact of exposure to new knowledge through fine-tuning on
tendency of the fine-tuned model to hallucinate.

**Quantifying knowledge in LLMs.** **S** li **CK** can
be seen as a confidence elicitation method for the
ground truth label ( _M_ _knows_ ( _q, a_ ) if it is confident
that _a_ is the answer to _q_ ). Existing work derive cali
brated confidence from LLMs by examining agree
ment across multiple samples ( Kuhn et al. , 2023 ;
Manakul et al. , 2023 ; Tian et al. , 2023a ; Lyu et al. ,
2024 ), probing internal representations ( Azaria and
Mitchell ), eliciting verbalized probability ( , 2023 ; Burns et al. Tian et al. , 2022 , 2023b ) or direct
prompting ( Kadavath et al. , 2022 ). Kadavath et al.
also trained a separate P(IK) model to predict if
the LLM knows the answer to _q_ . The label for
P(IK) was approximated by the fraction of correct
sampled answers, which is conceptually aligned
with _P_ `Correct` (§ 3 ). A key difference is that we also
define the **S** li **CK** categories, and provide evidence
that we capture meaningful and useful categories.

### 9 ### Conclusion

We study the impact of integrating new factual
knowledge through fine-tuning on the model’s ten
dency to hallucinate. We first propose **S** li **CK** , a
categorization of facts w.r.t. LLM’s knowledge.
We then design a controlled study where we iso
late the impact of new knowledge and rigorously
evaluate its effects. We provide multiple insights
on the fine-tuning dynamics, with the following key
findings: (1) Acquiring new knowledge via super
vised fine-tuning is correlated with hallucinations
w.r.t. pre-existing knowledge. (2) LLMs struggle to
integrate new knowledge through fine-tuning and
mostly learn to use their pre-existing knowledge.


-----


### 10 ### Limitations

Our experiments were conducted using a single
LLM, and thus it is unclear whether results will
vary with different LLMs. Having said that, our
study is extremely compute-heavy and thus challenging to replicate on multiple LLMs: First, our
fine-tuning is compute-heavy as its runs are very
long as we wanted to analyze the behavior during
different stages of fine-tuning (including the over
fitting stages). Second, and most importantly, to
facilitate our study we needed to annotate a large
scale dataset w.r.t the **S** li **CK** categories. To derive
reliable conclusions, it was crucial to accurately
assess the model’s knowledge w.r.t. a single fine
tuning example. In our case we run 170 inference
steps per example, i.e., more than 15 _M_ inference
steps to categorize our full dataset.
In addition, since we focus on closed-book QA,
the practical implications from our study such as
filtering-out `Unknown` fine-tuning examples still re
quire validation in settings involving long-form
text generation. To filter-out examples that introduce new factual knowledge in long-form generation tasks, one would need to make adaptations
to **S** li **CK** and come up with an effective way to
compare the sampled answer with the ground-truth
to approximate _P_ `Correct` . We leave this for future
work. Long-form generation tasks introduce eval
uation challenges, leading to a wide adoption of
LLM-based evaluations. Our choice to focus explicitly on closed book QA facilitates more accu
rate evaluation that enhances the reliability of our
findings.
Lastly, we did not test the effect of adding additional fine-tuning examples from diverse tasks
into the fine-tuning mixture. While this could
more closely approximate a typical instruction fine
tuning scenario, such dataset extension may intro
duce new factual knowledge in an uncontrollable
way, which will limit our findings.

### 11 ### Acknowledgments

We would like to thank Ori Ram, Uri Shaham, Alon
Jacovi, Mor Ventura, Yochai Blau, Eyal Ben-David,
Avi Caciularu, Avinatan Hassidim and the mem
bers of Roi Reichart’s NLP group for reviewing the
paper draft and providing valuable feedback. Spe
cial thanks to Uri Shaham for assisting in setting
up the fine-tuning pipeline during the early stages
of our research.


### References

Badr AlKhamissi, Millicent Li, Asli Celikyilmaz, Mona
Diab, and Marjan Ghazvininejad. 2022. A review on
language models as knowledge bases. _arXiv preprint_
_arXiv:2204.06031_ .

Rohan Anil, Andrew M Dai, Orhan Firat, Melvin John
son, Dmitry Lepikhin, Alexandre Passos, Siamak
Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng
Chen, et al. 2023. Palm 2 technical report. _arXiv_
_preprint arXiv:2305.10403_ .

Amos Azaria and Tom Mitchell. 2023. The internal
state of an llm knows when its lying. _arXiv preprint_
_arXiv:2304.13734_ .

Collin Burns, Haotian Ye, Dan Klein, and Jacob Stein
hardt. 2022. Discovering latent knowledge in language models without supervision. _arXiv preprint_
_arXiv:2212.03827_ .

Roi Cohen, Mor Geva, Jonathan Berant, and Amir
Globerson. 2023. [Crawling the internal knowledge](https://doi.org/10.18653/v1/2023.findings-eacl.139)
[base of language models](https://doi.org/10.18653/v1/2023.findings-eacl.139) . In _Findings of the Asso_
_ciation for Computational Linguistics: EACL 2023_ ,
pages 1856–1869, Dubrovnik, Croatia. Association
for Computational Linguistics.

Leo Gao. 2021. [Behavior cloning is miscalibrated](https://www.alignmentforum.org/posts/BgoKdAzogxmgkuuAt/behavior-cloning-is-miscalibrated) . _AI_
_Alignment Forum_ .

Yoav Goldberg. 2023. [Reinforcement learning for lan](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)
[guage models](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81) .

Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang
Geng, Hao Liu, Pieter Abbeel, Sergey Levine, and
Dawn Song. 2023. The false promise of imitating
proprietary llms. _arXiv preprint arXiv:2305.15717_ .

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, et al. 2023.
A survey on hallucination in large language models:
Principles, taxonomy, challenges, and open questions.
_arXiv preprint arXiv:2311.05232_ .

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom
Henighan, Dawn Drain, Ethan Perez, Nicholas
Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli
Tran-Johnson, et al. 2022. Language models
(mostly) know what they know. _arXiv preprint_
_arXiv:2207.05221_ .

Ehsan Kamalloo, Nouha Dziri, Charles L. A. Clarke,
and Davood Rafiei. 2023. [Evaluating open-domain](https://doi.org/10.18653/V1/2023.ACL-LONG.307)
[question answering in the era of large language mod](https://doi.org/10.18653/V1/2023.ACL-LONG.307)
[els](https://doi.org/10.18653/V1/2023.ACL-LONG.307) . In _Proceedings of the 61st Annual Meeting of_
_the Association for Computational Linguistics (Vol_
_ume 1: Long Papers), ACL 2023, Toronto, Canada,_
_July 9-14, 2023_ , pages 5591–5606. Association for
Computational Linguistics.

Katie Kang, Eric Wallace, Claire Tomlin, Aviral Ku
mar, and Sergey Levine. 2024. Unfamiliar finetuning
examples control how language models hallucinate.
_arXiv preprint arXiv:2403.05612_ .


-----


Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. 2023.
Semantic uncertainty: Linguistic invariances for un
certainty estimation in natural language generation.
_arXiv preprint arXiv:2302.09664_ .

Yoonsang Lee, Pranav Atreya, Xi Ye, and Eunsol
Choi. 2023. Crafting in-context examples according to lms’ parametric knowledge. _arXiv preprint_
_arXiv:2311.09579_ .

Bill Yuchen Lin, Abhilasha Ravichander, Ximing Lu,
Nouha Dziri, Melanie Sclar, Khyathi Chandu, Chan
dra Bhagavatula, and Yejin Choi. 2023. [The unlock](http://arxiv.org/abs/2312.01552)
[ing spell on base llms: Rethinking alignment via](http://arxiv.org/abs/2312.01552)
[in-context learning](http://arxiv.org/abs/2312.01552) . _ArXiv preprint_ .

Qing Lyu, Kumar Shridhar, Chaitanya Malaviya,
Li Zhang, Yanai Elazar, Niket Tandon, Marianna Apidianaki, Mrinmaya Sachan, and Chris
Callison-Burch. 2024. Calibrating large language
models with sample consistency. _arXiv preprint_
_arXiv:2402.13904_ .

Potsawee Manakul, Adian Liusie, and Mark JF Gales.
2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language
models. _arXiv preprint arXiv:2303.08896_ .

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and
Hannaneh Hajishirzi. 2022. [Cross-task generaliza](https://doi.org/10.18653/v1/2022.acl-long.244)
[tion via natural language crowdsourcing instructions](https://doi.org/10.18653/v1/2022.acl-long.244)
In _Proceedings of the 60th Annual Meeting of the_
_Association for Computational Linguistics (Volume_
_1: Long Papers)_ , pages 3470–3487, Dublin, Ireland.
Association for Computational Linguistics.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida,
Carroll L. Wainwright, Pamela Mishkin, Chong
Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray,
John Schulman, Jacob Hilton, Fraser Kelton, Luke
Miller, Maddie Simens, Amanda Askell, Peter Welin
der, Paul F. Christiano, Jan Leike, and Ryan Lowe.
2022.
Training language models to follow instructions with human feedback . In _Advances in Neural_
_Information Processing Systems 35: Annual Confer_
_ence on Neural Information Processing Systems 2022,_
_NeurIPS 2022, New Orleans, LA, USA, November 28_
_- December 9, 2022_ .

Fabio Petroni, Tim Rocktäschel, Sebastian Riedel,
Patrick Lewis, Anton Bakhtin, Yuxiang Wu, and
Alexander Miller. 2019. [Language models as knowl](https://doi.org/10.18653/v1/D19-1250)
[edge bases?](https://doi.org/10.18653/v1/D19-1250) In _Proceedings of the 2019 Confer_
_ence on Empirical Methods in Natural Language Pro_
_cessing and the 9th International Joint Conference_
_on Natural Language Processing (EMNLP-IJCNLP)_ ,
pages 2463–2473, Hong Kong, China. Association
for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christo
pher D Manning, Stefano Ermon, and Chelsea Finn.
2024. Direct preference optimization: Your language
model is secretly a reward model. _Advances in Neu_
_ral Information Processing Systems_ , 36.


Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. [SQuAD: 100,000+ questions for](https://doi.org/10.18653/v1/D16-1264)
[machine comprehension of text](https://doi.org/10.18653/v1/D16-1264) . In _Proceedings of_
_the 2016 Conference on Empirical Methods in Natu_
_ral Language Processing_ , pages 2383–2392, Austin,
Texas. Association for Computational Linguistics.

Ohad Rubin, Jonathan Herzig, and Jonathan Berant.
2022. [Learning to retrieve prompts for in-context](https://doi.org/10.18653/v1/2022.naacl-main.191)
[learning](https://doi.org/10.18653/v1/2022.naacl-main.191) . In _Proceedings of the 2022 Conference of_
_the North American Chapter of the Association for_
_Computational Linguistics: Human Language Tech_
_nologies_ , pages 2655–2671, Seattle, United States.
Association for Computational Linguistics.

John Schulman. 2023. [Reinforcement learning from](https://www.youtube.com/watch?v=hhiLw5Q_UFg&ab_channel=BerkeleyEECS)
[human feedback: Progress and challenges](https://www.youtube.com/watch?v=hhiLw5Q_UFg&ab_channel=BerkeleyEECS) .

Christopher Sciavolino, Zexuan Zhong, Jinhyuk Lee,
and Danqi Chen. 2021.
Simple entity-centric questions challenge dense retrievers . In _Proceedings_
_of the 2021 Conference on Empirical Methods in_
_Natural Language Processing, EMNLP 2021, Vir_
_tual Event / Punta Cana, Dominican Republic, 7-11_
_November, 2021_ , pages 6138–6148. Association for
Computational Linguistics.

Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D Manning, and Chelsea Finn. 2023a. Fine
tuning language models for factuality. _arXiv preprint_
_arXiv:2311.08401_ .

Katherine Tian, Eric Mitchell, Allan Zhou, Archit
Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn,
and Christopher D Manning. 2023b. Just ask for cali
bration: Strategies for eliciting calibrated confidence
scores from language models fine-tuned with human
feedback. _arXiv preprint arXiv:2305.14975_ .

Denny Vrandeˇ ci´ c and Markus Krötzsch. 2014. [Wiki](https://doi.org/10.1145/2629489)
[data: a free collaborative knowledgebase](https://doi.org/10.1145/2629489) . _Commun._
_ACM_ , 57(10):78–85.


Cunxiang Wang, Sirui Cheng, Qipeng Guo, Yuanhao
Yue, Bowen Ding, Zhikun Xu, Yidong Wang, Xiangkun Hu, Zheng Zhang, and Yue Zhang. 2023.

[Evaluating open-qa evaluation](http://papers.nips.cc/paper_files/paper/2023/hash/f323d594aa5d2c68154433a131c07959-Abstract-Datasets_and_Benchmarks.html) . In _Advances in Neu_
_ral Information Processing Systems 36: Annual Con_
_ference on Neural Information Processing Systems_
_2023, NeurIPS 2023, New Orleans, LA, USA, Decem_
_ber 10 - 16, 2023_ .

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin
Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. 2022. [Finetuned](https://openreview.net/forum?id=gEZrGCozdqR)
[language models are zero-shot learners](https://openreview.net/forum?id=gEZrGCozdqR) . In _The Tenth_
_International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022_ .
OpenReview.net.


Xunjian Yin, Baizhou Huang, and Xiaojun Wan. 2023.

[ALCUNA: Large language models meet new knowl](https://doi.org/10.18653/v1/2023.emnlp-main.87)
[edge](https://doi.org/10.18653/v1/2023.emnlp-main.87) . In _Proceedings of the 2023 Conference on_
_Empirical Methods in Natural Language Processing_ ,
pages 1397–1414, Singapore. Association for Com
putational Linguistics.


-----


Gal Yona, Roee Aharoni, and Mor Geva. 2024. Nar
rowing the knowledge evaluation gap: Open-domain
question answering with multi-granularity answers.
_arXiv preprint arXiv:2401.04695_ .

Hanning Zhang, Shizhe Diao, Yong Lin, Yi R Fung,
Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji,
and Tong Zhang. 2023. R-tuning: Teaching large
language models to refuse unknown questions. _arXiv_
_preprint arXiv:2311.09677_ .

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer,
Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping
Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis,
Luke Zettlemoyer, and Omer Levy. 2023. [LIMA:](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html)
[less is more for alignment](http://papers.nips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html) . In _Advances in Neural_
_Information Processing Systems 36: Annual Confer_
_ence on Neural Information Processing Systems 2023,_
_NeurIPS 2023, New Orleans, LA, USA, December 10_
_- 16, 2023_ .


-----


### A ### Data Preprocessing

This section expands § 2 with additional details
about our data preprocessing steps. The E NTI -

TY Q UESTIONS dataset ( Sciavolino et al. , 2021 ) con
sists of train, development and test splits and spans
24 relations. Our train, development and test sets
are curated based on the original splits from E NTI -

TY Q UESTIONS . However, we use only 12 relations,
since we wanted to reserve some relations for out
of-distribution test set. To avoid cherry-picking, the
12 relations used in our train, development and test
sets are randomly sampled. The resulting relations
are presented in Tables 3 and 4 .
We reserved the remaining 12 relations for out
of-distribution test set. However, we found that in
those 12 reserved relations, 5 were too similar to
some of the relations that we train on (Table 3 ),
thus we suspected that this could lead to a test set
that is not truly out-of-distribution. To address that,
we filtered out those relations and were left with
7 relations for our-of-distribution. Specifically we
filtered-out the following relations:

-  P276 was filtered out since it directly
overlaps with P131 since for both relations the question in E NTITY Q UESTIONS is
of the form _“Where is [E] located?”_ .
P276 stands for “location” ( [https://www.](https://www.wikidata.org/wiki/Property:P276)
[wikidata.org/wiki/Property:P276](https://www.wikidata.org/wiki/Property:P276) ) and
P131 stands for “located in the administrative
territorial entity” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P131)
[org/wiki/Property:P131](https://www.wikidata.org/wiki/Property:P131) ).

-  P20 for which the question template is _“Where_
_did [E] die?”_ was filtered out since it may
require a knowledge that is related to P19
for which the question template is _“Where_
_was [E] born?”_ . P20 stands for “place of
death” ( [https://www.wikidata.org/wiki/](https://www.wikidata.org/wiki/Property:P20)
[Property:P20](https://www.wikidata.org/wiki/Property:P20) ) and P19 stands for “place of
birth” ( [https://www.wikidata.org/wiki/](https://www.wikidata.org/wiki/Property:P19)
[Property:P19](https://www.wikidata.org/wiki/Property:P19) ).

-  P106 for which the question template is
_“What kind of work does [E] do?”_ was filtered
out since it may require a knowledge that is re
lated to P800 for which the question template
is _“What is [E] famous for?”_ . P106 stands
for “occupation” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P106)
[org/wiki/Property:P106](https://www.wikidata.org/wiki/Property:P106) ) and P800 stands
for “notable work” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P800)
[org/wiki/Property:P800](https://www.wikidata.org/wiki/Property:P800) ).



-  P413 for which the question template is
_“What position does [E] play?”_ was filtered out since it may require a knowledge that is related to P800 for which the
question template is _“What is [E] famous_
_for?”_ . P413 stands for “position played on
team / speciality” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P413)
[org/wiki/Property:P413](https://www.wikidata.org/wiki/Property:P413) ) and P800 stands
for “notable work” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P800)
[org/wiki/Property:P800](https://www.wikidata.org/wiki/Property:P800) ).

-  P159 for which the question template is
_“Where is the headquarters of [E]?”_ was
filtered out since it may require a knowledge that is related to P36 for which the
question template is _“What is the capital of [E]?”_ . P159 stands for “head
quarters location” ( [https://www.wikidata.](https://www.wikidata.org/wiki/Property:P159)
[org/wiki/Property:P159](https://www.wikidata.org/wiki/Property:P159) ) and P36 stands
for “capital” ( [https://www.wikidata.org/](https://www.wikidata.org/wiki/Property:P36)
[wiki/Property:P36](https://www.wikidata.org/wiki/Property:P36) ).

The 7 relations used for out-of-distribution test set
are presented in Table 5 .
Lastly, we perform two additional filtering steps:
(1) To simplify the process of categorizing the ex
amples w.r.t. _M_ ’s knowledge (§ 3 ), we filter-out
examples with more than 1 correct answer. 10 (2)
We make sure that no subjects or objects overlap
between the train and test sets, 11 by filtering-out
overlapping examples from the train set. 12

### B ### Test performance as Proxy for Hallucinations

We now detail the connection between the test per
formance in our setting and hallucinations. In our
study, poorer performance of a fine-tuned model
_M_ _D_ 1 compared to another fine-tuned model _M_ _D_ 2
on the test set can be attributed to a higher rate of
hallucinations in _M_ _D_ 1 , relative to its pre-existing
knowledge, due to the following explanation.
The test set can be conceptually divided into two
types of questions. First, there are questions with
answers that are unknown to _M_ . Those questions
will remain unknown post fine-tuning, as we make
sure that the training set is disjoint from the test

10 4 _._ 2% and 3 _._ 9% of the E NTITY Q UESTIONS train and test
set respectively.
11 For example, the subject _“_ **_Bruce Smith_** _”_ appears with
2 different relations ( _P_ 106 and _P_ 413 ) yielding 2 examples:
( _“What kind of work does_ **_Bruce Smith_** _do?”_ , _“poet”_ ) and
( _“Where was_ **_Bruce Smith_** _born?”_ , _“Faribault”_ ).
12 2 _._ 1% of the E NTITY Q UESTIONS train set.


-----


relation question template `HighlyKnown` `MaybeKnown` `WeaklyKnown` `Unknown` Total Min

P131 Where is [E] located? 553 2529 1493 3071 7646 553
P136 What type of music does [E] play? 236 3410 1892 1978 7516 236
P17 Which country is [E] located in? 4387 2628 511 364 7890 364
P19 Where was [E] born? 369 1884 1498 4170 7921 369
P26 Who is [E] married to? 1609 1503 1087 3257 7456 1087
P264 What music label is [E] represented by? 206 1444 1854 3820 7324 206
P36 What is the capital of [E]? 4160 1634 449 572 6815 449
P40 Who is [E]’s child? 692 1467 1271 2680 6110 692
P495 Which country was [E] created in? 5459 1101 408 706 7674 408
P69 Where was [E] educated? 233 1126 1712 3650 6721 233
P740 Where was [E] founded? 1323 1618 1428 2902 7271 1323
P800 What is [E] famous for? 301 330 222 503 1356 222

TOTAL -19528 20674 13825 27673 81700 6142

Table 3: Train set statistics.

relation question template `HighlyKnown` `MaybeKnown` `WeaklyKnown` `Unknown` Total

P131 Where is [E] located? 57 362 158 388 965
P136 What type of music does [E] play? 6 432 248 281 967
P17 Which country is [E] located in? 448 432 65 51 996
P19 Where was [E] born? 107 148 243 501 999
P26 Who is [E] married to? 177 238 158 378 951
P264 What music label is [E] represented by? 47 157 268 486 958
P36 What is the capital of [E]? 580 152 62 86 880
P40 Who is [E]’s child? 99 191 167 344 801
P495 Which country was [E] created in? 699 147 51 96 993
P69 Where was [E] educated? 27 145 227 441 840
P740 Where was [E] founded? 182 245 181 334 942
P800 What is [E] famous for? 35 50 28 76 189

TOTAL -2464 2699 1856 3462 10481

Table 4: In-distribution test set statistics.


set (§ A ). This means that both _M_ _D_ 1 and _M_ _D_ 2 will
fail to answer these questions. Thus, the test performance difference between _M_ _D_ 1 and _M_ _D_ 2 is mostly
attributed to the second type of questions: ones that
are known to _M_ , i.e. _M_ can answer them correctly
since it posses the relevant knowledge. Thus, _M_ _D_ 1
and _M_ _D_ 2 must rely on their pre-existing knowledge
to answer such questions, and a lower performance
on such question can be only categorized as an
hallucination w.r.t. pre-existing knowledge.

### C ### P `Correct` ### Approximation

This section expands § 3 with additional details
about our _P_ `Correct` approximation. In our study
we approximate _P_ `Correct` ( _q, a_ ; _M, T_ ) based on the
fraction of correct answers to _q_ sampled from _M_ .
We begin with randomly sampling _N_ ex distinct _k_
shot exemplars for each relation in our dataset (§ A ).
Then, to approximate _P_ `Correct` ( _q, a_ ; _M, T_ ) , we use
_M_ to generate answers to _q_ using each of the _N_ ex
exemplars from the relation corresponding to _q_ .


We first use temperature sampling with _T_ = 0 _._ 5
to sample _N_ sample answers for each of the _N_ ex ex
emplars. _P_ `Correct` ( _q, a_ ; _M, T >_ 0) is then approxi
mated by the fraction of correct answers from the
total of _N_ ex _·_ _N_ sample predictions. We also generate
the greedy decoding prediction ( _T_ = 0 ) for each
of the _N_ ex exemplars. _P_ `Correct` ( _q, a_ ; _M, T_ = 0) is
then approximated by the fraction of correct an
13
swers from the total of _N_ ex predictions.

We use _k_ = 4 in our study, simply since we
found it enough for _M_ to output answers in the
correct format. We use _N_ ex = 10 and _N_ sample =
16 . The _N_ sample = 16 samples using _T_ = 0 _._ 5 are
sampled from Top 40.
The _k_ exemplars are sampled from the develop
ment split. We sample _N_ ex different samples since
we found that even when the few-shot exemplars
are sampled per-relation, their exact choice still
affects the prediction. In § 6 and Figure 5 we show
evidence that this also improves the quality of our


We first use temperature sampling with _T_ = 0 _._ 5
to sample _N_ sample answers for each of the _N_ ex ex
emplars. _P_ `Correct` ( _q, a_ ; _M, T >_ 0) is then approxi
mated by the fraction of correct answers from the
total of _N_ ex _·_ _N_ sample predictions. We also generate
the greedy decoding prediction ( _T_ = 0 ) for each
of the _N_ ex exemplars. _P_ `Correct` ( _q, a_ ; _M, T_ = 0) is
then approximated by the fraction of correct an
13
swers from the total of _N_ ex predictions.


13 Since we can only have one greedy prediction for every
k-shot exemplars.


-----


relation question template `HighlyKnown` `MaybeKnown` `WeaklyKnown` `Unknown` Total

P127 Who owns [E]? 125 383 168 314 990
P50 Who is the author of [E]? 287 193 115 372 967
P407 Which language was [E] written in? 366 153 59 45 623
P176 Which company is [E] produced by? 289 277 181 225 972
P170 Who was [E] created by? 142 284 120 304 850
P175 Who performed [E]? 94 120 103 663 980
P112 Who founded [E]? 134 116 76 140 466

TOTAL -1437 1526 822 2063 5848

Table 5: Out-of-distribution test set statistics.


Wrong Answer Paraphrase Higher Granularity Lower Granularity

90% 6% 2% 2%

Table 6: Error Analysis of 100 Predictions of the Pre
trained Model, for Which Exact Match is False.

categories.
Below is an example of our 4-shot prompt for
mat, from real example from E NTITY Q UESTIONS with

14
the relation _P_ 106 representing occupation. The
question in this case is _“What kind of work does_
_Ron Konopka do?”_ and the ground truth asnwer is
_“geneticist”_ .

![2405.05904v1.pdf-13-0.png](2405.05904v1.pdf-13-0.png)

To decide whether a sampled answer is correct,
we use the Exact Match (EM) metric to compare it
with the ground truth answer. The main advantage
in this choice is that when EM is True, we know
that the answer is correct for 100% . The main
potential risk associated with this choice is that we
may wrongly classify answers as incorrect due to
paraphrases or answers with different granularity
levels ( Wang et al. , 2023 ; Kamalloo et al. , 2023 ;

Yona et al. , 2024 )). To address this, we perform
an **error analysis** on 100 predictions for which
EM was False. We randomly sample 50 greedy
predictions ( _T_ = 0 ) and 50 samples ( _T_ = 0 _._ 5 ).
The results are in Table 6 . This analysis suggest
that in 90% of the cases where EM is False, the
predicted answer is indeed incorrect. Which is a
reasonable performance for our purpose, especially

14 [https://www.wikidata.org/wiki/Property:P106](https://www.wikidata.org/wiki/Property:P106)


considering that when EM is True the answer is
100% correct (which will never happen with any
other metric).

### D ### Data Annotation

we first calculate _P_ `Correct` ( _q, a_ ; _M, T_ = 0) and
_P_ `Correct` ( _q, a_ ; _M, T >_ 0) for each ( _q, a_ ) pair in
our preprocessed dataset (§ 2 and § A ), using our
using our _P_ `Correct` ( ) approximation (§ 3 and § C ).

_·_
We then use these values to categorize each ( _q, a_ )
pair into one of our four categories (§ and Figure ). We provide the full statistics of the categories on the train and test set, as well as the 2 3
out-of-distribution test set in Tables 3 , 4 and 5 .

### E ### More Details About or Variants of the Fine-tuning dataset ( D ).

In § 4 we examine the effect of new knowledge in
the fine-tuning dataset _D_ on the performance of
_M_ _D_ , by varying the proportion of `Unknown` exam
ples in _D_ . When we create variants of _D_ with
exactly _X_ % of `Unknown` and (100 _−_ _X_ )% `Known`
examples, we make sure that the relation distribu
tion remains consistent. To achieve that we sample
_X_ % of `Unknown` _from each relation_ .
In § 5 we create single-category variants of _D_ .
Since we want to work with a fixed _|_ _D_ _|_ across
all variants, we want to make sure that we have
_|_ _D_ _|_ examples from each category. To ensure this,
we measure the size of the smallest category in
each relation (see the “Min” column in Table 3 )
and define _|_ _D_ _|_ as their sum. In other words, for
each relation we calculate the size of the smallest
category and sum these values. This leads to _|_ _D_ _|_ =
6142 , as illustrated by the last column in Table 3 .
More formally, for each relation r in the training
split, and for each category CAT from our 4 **S** li **CK**
categories, we define we CAT r to be the examples
from category CAT and relation r . Consequently


-----


we treated the `Known` categories collectively. For
reference we also include the plot with the full
per-category breakdown in Figure 6 .

### G ### Linear Model

In § we use a linear model (Equation ( 4.4 1 )) that predicts that test accuracy and the and § 4.5
out-of-distribution test accuracy. We estimate the
parameters of this linear model based on results
from all our variants of _D_ used in § 4 . For all these
variants, we measure the test accuracy and the num
ber of `Known` and `Unknown` fine-tuning examples
that _M_ fits during different fine-tuning stages. This
way we collect a dataset with examples of the form
( _Accuracy, N_ Kn _, N_ Unk ) , which we use to fit a lin
ear regression model.

### H ### Out-of-distribution (OOD) Evaluation

In § results. In these experiments we simply used our 4.5 we discuss _out-of-distribution (OOD)_
OOD test set consisting of 7 relations unseen dur
ing fine-tuning (see § A ). When we perform the
analysis discussed in § 4.1 and § 4.2 , we addition
ally evaluated the models on the OOD test set. For
completeness, we add here Figure 7 , which is the
out-of-distribution version of Figure 3 . Figure 7a
presents the OOD test performance as a function
of % of `Unknown` examples in _D_ for different fine
tuning duration. The corresponding _in-distribution_
results (Figure 3a ) were discussed in § 4.1 . Fig
ure 7b presents the OOD test performance for the
ablation where we filter-out `Unknown` fine-tuning
examples. The corresponding results (Figure 3b ) were discussed in § _in-distribution_ 4.2 . We no
tice that similar trends, just with a smaller overall
magnitude of the performance drop, up to 6 points
drop compared to up to 14 for in-distribution. This
smaller drop magnitude is also reflected in smaller
values of _|_ _β_ ukn _|_ and _|_ _β_ kn _|_ (Table 1 ).

### I ### Statistic Significance Tests

In § 5 we present Table 2 . As mentioned in the
caption, we perform statistic significance tests for
each column. To this end we compare all the values
to the maximal value in this column.
For each subset of the test set, we randomly
shuffle all the examples in it, split them up into 100
approximately equally sized subsets, and compute
accuracy for each of them for all the models of
interest. We then apply paired-sample t-test with
_p <_ 0 _._ 05 and _p <_ 0 _._ 01 .


![2405.05904v1.pdf-14-1.png](2405.05904v1.pdf-14-1.png)

Figure 6: Training accuracy as a function of fine-tuning
duration, evaluated on the variant with 50% `Unknown`
fine-tuning examples. For reference, we also include
the accuracy on the development set, accompanied by a
zoom-in plot within a narrower range, to provide a more
visible and clear view.

size ( CAT r ) is the number of the examples in CAT r .
For example size ( `HighlyKnown` P131 ) = 553 (see
Table 3 ). We then define:


CAT _∈{_
```
HighlyKnown

```
`MaybeKnown` _,_
```
WeaklyKnown
Unknown

```
_}_


_|_ _D_ _|_ =


min
_r_ _∈_ _R_ Train


size ( _CAT_ _r_ )
_|_


where R Train are the 12 relations from the training
set.
Below is an example of our data format in the
train, development and test sets, from real example
from E NTITY Q UESTIONS with the relation _P_ 106 rep
resenting occupation. 15 The question in this case is
_“What kind of work does Ron Konopka do?”_ and the
ground truth asnwer is _“geneticist”_ .

![2405.05904v1.pdf-14-0.png](2405.05904v1.pdf-14-0.png)

### F ### Train Accuracy on Different `Known` Categories

In § 4.3 we analyze the fine-tuning dynamic and
present the training accuracy as function of the
fine-tuning duration in Figure 1 . For simplicity

15 [https://www.wikidata.org/wiki/Property:P106](https://www.wikidata.org/wiki/Property:P106)


-----


![2405.05904v1.pdf-15-1.png](2405.05904v1.pdf-15-1.png)

![2405.05904v1.pdf-15-2.png](2405.05904v1.pdf-15-2.png)

(a) (b)

Figure 7: Performance on the _out-of-distribution (OOD)_ test set as a function of the % of `Unknown` examples in the
fine-tuning dataset _D_ . This plot is the OOD version of Figure 3 . Everything is similar to Figure 3 , except that y-axis
is the accuracy on the OOD test set. We note that **_the development set did not change (not OOD)_** , thus it does not
necessarily reflects the optimal stopping point for OOD.

EARLY _ STOP C ONVERGENCE
```
       Full Hkn Mkn Wkn Unk Full Hkn Mkn Wkn Unk

```

_D_ `HighlyKnown` 40.5 _∗∗_ **98.7** 60.1 _∗∗_ 9.0 _∗∗_ 0.6 _∗∗_ 40.0 _∗∗_ **98.4** 58.8 _∗∗_ 8.5 _∗∗_ 0.7 _∗∗_


_D_ `MaybeKnown` **43.6** **98.4** **69.9** 12.1 _∗∗_ 1.0 _∗∗_ **43.2** 97.5 _∗_ **68.2** 12.9 _∗∗_ 1.3 _∗∗_


_D_ `WeaklyKnown` 39.2 _∗∗_ 95.0 _∗∗_ 59.2 _∗∗_ 8.6 _∗∗_ 0.4 _∗∗_ 35.4 _∗∗_ 73.5 _∗∗_ 55.8 _∗∗_ **17.2** 2.2 _∗∗_


_D_ `Unknown` 37.5 _∗∗_ 95.6 _∗∗_ 52.9 _∗∗_ 6.5 _∗∗_ 0.6 _∗∗_ 25.8 _∗∗_ 55.8 _∗∗_ 36.6 _∗∗_ 12.2 _∗∗_ **3.2**

_D_ `Natural` **43.5** 98.0 _∗_ 67.6 _∗∗_ **14.1** **1.8** 41.8 _∗∗_ 95.5 _∗∗_ 61.7 _∗∗_ 14.8 _∗∗_ 2.5 _∗_

Table 7: A copy of Table 2 with detailed notation of the statistic significant test results. In each column, statistically
significant differences from the best result are indicated using _∗_ and _∗∗_ for _p <_ 0 _._ 05 and _p <_ 0 _._ 01 respectively.


In Table 2 , the best result is in bold, as well as all
the results with statistically not significant differ
ence from the best with _p <_ 0 _._ 05 . We additionally
include a copy of Table 2 where all the statistical
tests outcomes are annotated, see Table 7 . We can
see that in almost all cases the difference is statis
tically significant with _p <_ 0 _._ 01 , except two cases
where it is only with _p <_ 0 _._ 05 ( _D_ `Natural` `Unk` and
_D_ `MaybeKnown` `Mkn` ).
Since we also discuss “horizontal” comparisons,
where we compare EARLY _ STOP to C ONVERGENCE ,
we additionally run significance tests (not anno
tated in Table 2 ) for _All_ , comparing EARLY _ STOP to

C ONVERGENCE . The difference for _D_ `MaybeKnown` was
not statistically significant while for all others (in
cluding _D_ `Natural` ) it was significant with _p <_ 0 _._ 01 .

### J ### The P(True) Case Study

In § 6 we used the P(True) metric from Kadavath
et al. ( 2022 ) as a case study for comparison. In
Figure 5 we compare our `Unknown` category vs
classifying as `Unknown` based on a threshold of


P(True). We calculated P(True) for every ( _q, a_ )
pair in the test set using Kadavath et al. ( 2022 )’s
prompt:

![2405.05904v1.pdf-15-0.png](2405.05904v1.pdf-15-0.png)

We then treated ( _q, a_ ) pairs with P(True) below
a threshold as `Unknown` . We experimented with
each possible threshold _T_ in [0 _,_ 1] . For each thresh
old _T_ we then measured (1) how many examples
were classified as `Unknown` out of the test set, (2)
what was the accuracy on these examples after
fine-tuning. We plot the results in Figure 5 , where
P(True) is represented with the **yellow line** and our
`Unknown` is represented with the **blue circle** . As
discussed in § C , it was approximated using 10 def
ferent samples of 4-shot exemplars ( _N_ ex = 10 ). We
also check smaller values of _N_ ex and plot the results
with the **blue line** . The accuracy after fine-tuning


-----


EARLY _ STOP C ONVERGENCE

Accuracy % Answered Accuracy % Answered

_D_ 43.0 100.0 38.8 100.0

_D_ `IDK` 61.8 58.7 61.8 55.6

Table 8: Results of our initial experiment where the label
of the `Unknown` fine-tuning examples is replaced with
_“I don’t know”_ . _D_ in this case is the variant with 50%
`Known` and 50% `Unknown` . _D_ `IDK` is the variant where all
the 50% `Unknown` fine-tuning examples were re-labeled
with _“I don’t know”_ . The accuracy is measured on the
subset of the test questions that were answered, i.e. _M_ _D_
did not respond with _“I don’t know”_ .


Specifically, the accuracy for _D_ `IDK` remains 61 _._ 8
for both EARLY _ STOP and C ONVERGENCE , with a small
decrease on the number of willingly answered ques
tions ( 58 _._ 7 _→_ 55 _._ 6 )


for all the results is measured after fine-tuning with
_D_ `Natural` (§ 5 ).

### K ### Re-labeling `Unknown` ### Fine-tuning Example with an Uncertainty Expression: Initial Experiment

In this work we showed that fitting `Unknown` fine
tuning examples negatively affects the test perfor
mance. However, this negative effect manifests as
a form of _overfitting_ . From practical perspective,
we showed that we can mitigate overfitting by ei
ther using early-stopping or filtering-out `Unknown`
examples from the fine-tuning dataset.
We now perform a preliminary experiment
where check whether fine-tuning the model to ab
stain from `Unknown` examples can also be a poten
tial mitigation. Specifically, we replace the label of
the `Unknown`
fine-tuning examples with the expression _“I don’t know”_ and test whether this mitigates
the observed overfitting.
Table 8 presents the % of the test questions that
were answered (i.e. _M_ _D_ did not respond with _“I_
_don’t know”_ ) and the accuracy on those questions.
This experiment was conducted on the _D_ variant
with 50% `Unknown` . The first row is for the original
result with _D_ as a reference and the second row is
for the results with _D_ `IDK` , where the ground-truth
label of the 50% of the `Unknown` examples in _D_
was replaced with _“I don’t know”_
Consistent with the findings from previous work
( Zhang et al. , 2023 ), we observe an improved
accuracy on willingly answered test examples
(when comparing _D_ vs _D_ `IDK` ). When we compare

EARLY _ STOP vs C ONVERGENCE for _D_ we observe a
performance drop ( 43 _._ 0 _→_ 38 _._ 8 ) which illustrates
the overfitting effect. However, we observe that re
labeling the `Unknown` examples with uncertainty
expression seem to reduce the risk of overfitting.


-----

