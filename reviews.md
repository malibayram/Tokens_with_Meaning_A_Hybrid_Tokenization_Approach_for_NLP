Reviews:

-Reviewer 1:
The topic of linguistically informed tokenization has gained considerable attention in recent years,
particularly as large language models are increasingly applied to morphologically rich and low-resource
languages. This paper addresses an important and timely problem by proposing a hybrid tokenization
framework that integrates rule-based morphological analysis with BPE, using Turkish as the primary case
study. The proposed method is conceptually interesting and shows promise, especially in improving
morphological coherence and token-level linguistic alignment. However, the current version of the
manuscript suffers from substantial structural, methodological, and empirical limitations that prevent it
from meeting the standards of a publishable contribution. For these reasons, I recommend rejection in
its present form:

1. The Introduction currently presents detailed quantitative findings, which belong in the Results
   section. The opening should instead set up the problem, research questions, and high-level
   contributions.
2. Although the text claim support for morphologically rich languages, the study’s experiments and
   analysis are confined only to agglutinative systems (primarily Turkish) so the scope must be
   adjusted or additional language types (analytical, inflectional) included.
3. The Related Work survey focuses heavily on agglutinative contexts (Turkish, Finnish, Hungarian)
   while citing languages such as English and Chinese without specifying their morphological
   typology. The manuscript would benefit from explicitly identifying the typological classification of
   each mentioned language and incorporating tokenization research on other morphologically rich,
   low-resource inflectional languages. For example, a recent study introduces the Slovak
   Morphological Tokenizer [https://doi.org/10.7717/peerj-cs.2465], which, similarly to the
   approach proposed in this manuscript, preserves the integrity of root morphemes in individual
   tokens through a morphology-aware BPE framework.
4. The final paragraph of the Related Work section is problematic, as it shifts away from surveying
   prior literature and instead offers a summary and speculative suggestions for future research
   directions. Such content is not appropriate in a literature review and would be more suitably
   placed in the Conclusion or a dedicated Discussion section.
5. Section 3 (Methodology) begins with background on standard tokenizers rather than immediately
   describing the novel pipeline, and lacks clear sub-sectioning (e.g. Dictionary Construction,
   Encoding Algorithm, Decoding Process) to guide the reader through each component.
6. The Methodology section states that the root dictionary is constructed from high-frequency
   words extracted from large-scale Turkish corpora, yet no specific corpus names or sources are
   cited at this point in the text: „The root dictionary is constructed from high-frequency words
   extracted from large-scale Turkish corpora“. For transparency and reproducibility, the exact
   datasets used for root extraction should be clearly referenced where first mentioned.
7. The structure of the Methodology section is confusing and undermines clarity. The authors begin
   by describing the proposed morphological tokenization approach in overly general terms,
   including evaluative statements such as “the tokenizer achieves improved performance” and
   “the methodology is adaptable to other morphologically complex languages,” despite no results
   having yet been presented at that point. These claims are premature and should be reserved for
   the Results or Conclusion sections. Following this, the authors return to a more detailed
   explanation of the proposed system; however, at the point where the root dictionary is
   introduced, the source of the “large-scale corpora” used for extracting high-frequency Turkishwords is again omitted. Additionally, the process by which high-frequency words were extracted
   from these corpora is not described, leaving a significant methodological gap.
8. Without pseudocode or an algorithmic box detailing the hybrid tokenization procedure, the
   reader cannot fully understand or reproduce the proposed method; a precise step-by-step
   description is essential.
9. The handling of acronyms or runs of uppercase letters (for example word: „HTTPServer“) is not
   specified, despite the introduction of an <uppercase> token for single capital letters – this edge
   case should be addressed explicitly.
10. The Methodology is presented as one dense narrative; breaking it into titled subsections and
    supplementing with a diagram or table of core parameters would greatly improve readability.
11. The manuscript repeatedly uses the full term „Byte Pair Encoding“ even after introducing the
    abbreviation „BPE“. Once the abbreviation is defined, the full phrase should be avoided in
    subsequent occurrences to maintain clarity and conciseness.
12. The Methodology section inappropriately includes remarks about potential future applications of
    the proposed framework, which should be reserved for the Conclusion or Discussion section.
13. Table 1 reports metrics only for the proposed tokenizer; a consolidated comparison table
    including all baseline tokenizers is needed to transparently demonstrate relative performance.
    Furthermore, it is unclear whether the selected tokenizers are fully comparable, as many of them
    were not primarily trained on Turkish data. The authors should justify their choice of baselines
    and consider including tokenizers from Turkish-specific language models to ensure a fair and
    meaningful comparison.
14. The manuscript relies on reported correlations between TR %/Pure % and MMLU performance
    but does not include any direct evaluation on downstream tasks. To demonstrate the practical
    value of the proposed tokenization approach, the authors should pretrain a language model using
    their tokenizer and fine-tune it on at least one real-world downstream task. Without such
    evaluation, the claimed improvements remain purely theoretical.
15. The proposed tokenizer generates substantially more tokens (707 k vs. 434 k for aya-expanse).
    The impact on model inference speed, memory usage, and overall system efficiency is not
    analyzed. This trade-off should be characterized, or at least noted as a limitation.
16. All experiments target Turkish, yet the paper claims language independence; the authors should
    either present preliminary results on another language or clearly frame cross-linguistic
    applicability as future work and acknowledge it as a limitation.
17. The approach relies on a manually curated root dictionary (app. 22 000 Turkish roots). The
    authors should discuss the practical challenges of obtaining comparable resources for other
    languages and include this in the study’s limitations.

- Reviewer 2:
  This paper proposes a tokenization approach for morphologically
  complex languages such as  Finnish, Hungarian, Turkish and the like,
  in which  instead of being based on e.g. BPE or WordPieces methods,
  tokens are based on morphological structure with additional
  morphophonological abstractions to further reduce vocabulary size.
  Intuitively the motivation is compelling. The authors right at the
  outset claim that  "Evaluation on the TR-MMLU
  benchmark—a large-scale, Turkish-specific NLP benchmark—demonstrates that
  the proposed tokenizer achieves the highest Turkish Token Percentage (90.29%)
  and Pure Token Percentage (85.8%) among all tested models." which
  leads to some excitement and a heightened expectations -- which
  unfortunately are never to come!.

Maybe I am missing something but when one suggests a such a tokenization algorithm, one should show its
efficacy by showing how/if it improves the actual results in the tasks
of the benchmark. If the tasks show statistically significant gains
compared to competing tokenization methods while keeping other
variations (e.g. training data, training protocol etc.), one would
then believe that the tokenization proposed is superior.  But this is
a rather hard task.  One would need to pretrain LLMs using vocabulary
selected by the alternate tokenizer, compute embeddings for the
entries in the tokenization dictionary, and then run through the tasks
and compare.   As far as I can tell, the %'s provided and mentioned
above do not necessarily point to improvements in "performance of downstream tasks such as question answering, sentiment analysis,
and machine translation" although the authors indicate they
correlate.  Well, I would like to see the causal relationship for
this.

To be fair, the authors actually propose a so-called hybrid algorithm
"combining dictionary-based morphological analysis with Byte
Pair Encoding. While morphological segmentation ensures alignment with linguistic
units, BPE provides fallback coverage for unknown words, maintaining efficiency and
scalability in large corpora." So the hybrid component is actually a fall-back
unit.

There are numerous other issues with the proposed approach:

1. These days LLMs are almost all multi-lingual, being trained on
   data  literally  from 10s, if not hundreds of languages. While I
   sympathize with the authors' motivation that linguistically informed
   tokenization may be better for a single language, it is not clear how
   that can be applied in such a massive-multilingual case even with some
   hybrid support. Now you can
   certainly train a monolingual LLM for a specific language and use
   tokenizer like the one proposed, but you will be doing away with all
   the other useful functionality brought in by multilinguality --
   e.g. machine translation, and the like.

2. The Turkish-specific tokenization approach again seems to be a
   reasonably method to use until you realize that it is a bit
   haphazard.  First there is the issue of phonological abstractions:
   Once the method identifies morphemes with
   open vowels that differ in the front-back dimension {a, e}, then
   the morphemes (e.g. -ler/-lar)  can be abstracted into  e.g., -lAr contributing to
   the reduction in vocabulary! There is also some discussion how
   consonant assimilation cases can be handled. But I do not see any
   discussion of the vowels {ı, i, u, ü}  and there are many morphemes
   with such alternations.  In fact,  the example tokenizations on page 13
   are NOT displaying these abstract tokens so that is very confusing.

3. Further, there are many other consonant phenomena that are not even
   mentioned;  e.g., k/ğ alternations, consonant assimilations across
   morpheme boundaries, consonant gemination in words borrowed from
   mostly Arabic (e.g. ,  üssü, tıbbı). and the like. There are also many
   exceptions to harmony based on _written_ forms dues to mostly
   palatalization of preceding consonants.

4. What is also not clear is whether  the morpheme segmentations are on
   the surface form of the morphological structure (which I suspect the
   authors are doing) or over the lexical morpheme structure. Surface
   segmentations which are the same may actually correspond to different
   lexical segmentations (e.g., evinde would have two different lexical
   segmentations when used in "onun evinde" vs "senin evinde").
   Furthermore, the assumption of the longest matching root is also iffy
   if you are going to be aligned to the linguistic structure. For
   example, on page 13 why is okuma a token as clearly it is a derived
   nominal from a verb with a potential structure oku+mA

At this point I am nitpicking but the point is  if you are going to be  linguistically
informed you should go most of the way in.  I am also not going to
comment too much on the generation side of things, where things can
get rather tricky.

5. While the presentation is quite clear, there is one very bothersome
   convention that the authors use, citations as if they are sentence
   constituents; e.g.,

--"The morphological tokenizer introduced by [8] outperformed"   -- should
be something like "The morphological tokenizer introduced by First-Author et al., [8], outperformed.."

-- " [17] demonstrated that morphology-aware.." -- should be like
 " First-Author et al., [17],  demonstrated that morphology-aware..."

This is really bad style.

6. In the references, please find the actually refereed and published
   versions of arxiv entries.  You should use consistent word cases in
   paper titles; sometime all content words are capital initial (e.g.,
   [19]) sometime not (e.g., [16])

- Reviewer3:
  Key Results: 
  The manuscript proposes a 3-stage rule-based tokenization system to improve morphological segmentation for the Turkish language. The authors benchmark their system on the TR-MMLU dataset and compare with existing tokenizers of gemma-3, llama-3.2, qwen-2.5, phi-4, gpt-4o and aya-expense. The system reports significant improvements in Turkish Token Percentage (TR%) and Pure Token Percentage (Pure%) metrics, and includes qualitative examples illustrating over-segmentation and morphological boundary violations in baseline tokenizers.
  Clarity and Context:
  The abstract and introduction are well-written, clearly contextualizing the challenges of tokenization in agglutinative languages and motivating the need for morphological awareness. The paper effectively highlights limitations in standard BPE/WordPiece approaches, particularly their tendency to ignore morphological boundaries, and justifies the inclusion of whitespace and special character handling in tokenization.
  Validity: 
  While the proposed approach is relevant and the TR% and Pure% results are promising, several issues affect validity:
  Incomplete metrics: The reported results omit processing time and downstream task performance. Claims of Rust-based speed improvements are unsupported, and there is no evidence of impact on tasks such as MMLU performance, despite prior related work (arXiv:2502.07057) by the same authors including such measures.
  Limited benchmarks: Comparisons are restricted to general-purpose tokenizers trained largely on English-heavy corpora. No evaluation against specialized open-source Turkish tokenizers (e.g., Orbina, TurkCell) from the OpenTurkishLLM Leaderboard is presented.
  Metric dependency: TR% and Pure% may reflect properties of Turkish morphology and the TR-MMLU dataset rather than true tokenizer quality. For languages where many words are single morphemes, these metrics would be inflated.
  These gaps weaken claims about generalizability and real-world applicability.
  Originality and Significance: 
  The paper's overall segmentation algorithm, while original, doesn't present any unique idea or approach. For instance, space, punctuation, case, and unknown‑token handling are presented as contributions, but these are common in modern tokenizers. The addition of "uppercase token" to differentiate capitalized words is also explored previously here: https://ceur-ws.org/Vol-2829/paper2.pdf. Similarly, the idea of adding suffixes is also explored here: https://arxiv.org/abs/2307.07262v2 
  Furthermore, some design choices raise concerns:
  Treating frequent compounds (e.g., akarsu, çamaşırhane) as single tokens contradicts the paper’s stated goal of morpheme preservation and likely inflates Pure%.
  The reported improvements could be artifacts of dataset composition rather than true algorithmic advancement.
  Data and Methodology: 
  The TR-MMLU benchmark is a strong dataset contribution, but more examples demonstrating qualitative tokenization results would help illustrate improvements. I encourage the authors to share more examples from the data highlighting the qualitative results.
  The methodology is verbose and repetitive. Tokenization and decoding processes are described multiple times, diluting clarity. For instance, the paper first lists down the method from word segmentation to affix identification to BPE and root word analysis. Then the paper lists down decoding strategy and then follows it up with root word analysis, affix identification and BPE integration followed by encoding again. It may be suitable to break down the paper into major sections for readability. 
  The decoding pipeline is underspecified: how affixes recombine, and the rules for vowel deletion, consonant softening, and contraction, are left ambiguous. Parameter choices (e.g., 230 affixes, 10k BPE vocab size) are not justified or analyzed through ablations.
  Conclusions: 
  While results on TR% and Pure% are robust, claims about downstream NLP benefits are speculative without supporting metrics (accuracy, perplexity, or MMLU).
  Furthermore, the Turkish Token Percentage (TR%) and Pure Token Percentage (Pure%) metrics could be language dependent. The previous works by the same authors have explored them only in Turkish. Testing on other agglutinative languages (e.g., Finnish) or including downstream evaluations would strengthen generalizability and credibility. 
  References: 
  The authors provide ample references to existing works, and are well aware of the problems in existing methodologies. The paper omits comparison to current open-source Turkish tokenizers. Including these comparisons would provide stronger empirical grounding.
  Suggested Improvements:
  Report efficiency metrics: Include processing time (s) for all compared tokenizers to validate Rust speed claims.
  Add downstream evaluation: Report MMLU or other accuracy-based metrics, as done in prior related work (arXiv:2502.07057), to demonstrate real-world performance benefits.
  Benchmark against open-source Turkish tokenizers: Compare to Orbina, TurkCell, and other models from the OpenTurkishLLM Leaderboard.
  Clarify TR%/Pure% dependence: Test on another agglutinative language (e.g., Finnish) or multiple Turkish corpora to validate metric robustness.
  Revisit compound treatment: Justify or remove compounds as single tokens and re-run TR%/Pure% without this design choice.
  Streamline methodology: Consolidate into one algorithm box with encode/decode pseudocode and worked examples.
  Detail decoding strategy: Explicitly define rules for affix recombination, vowel deletion, consonantization, and contraction.
  Provide pipeline examples: Include tokenization outputs for sample Turkish words showing segmentation, affix mapping, and decoding.
  Expand result tables: Add token counts, vocab size, TR%, Pure%, processing time, and error breakdowns for each tokenizer.
  Contextualize novelty: Explicitly compare your contributions to related work (uppercase token, affix handling) and highlight distinct differences.

- Reviewer4:
  This paper tackles an important issue: how tokenizers may be poorly suited to less represented
  languages in NLP. This paper proposes a novel tokenization method that the authors argue
  better represents Turkish. However, I raise issues with the paper itself and the details of the
  work. I think overall the work needs significant revision and expansion of the experiments in
  order to be published. Based on the scope established in the paper, I believe a revised and
  significantly shortened version of the work would be more appropriately published as a short
  paper (4 pages) at an ACL venue. Therefore, I recommend a reject decision for this manuscript.
  This paper is written in a way that makes it difficult to evaluate the work. The introduction is too
  long. The introduction is unclear, contains some inaccuracies, and is repetitive. Material that
  should be in the introduction or related work sections is in the methodology section.
  ● Inaccurate representation of the literature:
  ○ In the related work section, the authors motivate the work with a small number of
  papers that argue that tokenization is important for morphologically rich
  languages (top of p. 5), but fail to cite any of the work that argues the opposite.
  See [1, 2, 3] and citations within.
  ○ Similarly, the authors claim that “morphologically rich languages such as Turkish,
  Finnish, and Hungarian are frequently segmented in ways that violate morphemic
  boundaries”. There is work, e.g. [1], which empirically shows the exact opposite.
  The authors should better represent the work that contradicts the claims in the
  paper.
  ○ Again, for the claim “While experimental results consistently show that
  morphology-aware tokenization improves efficiency and accuracy, most
  large-scale language models still rely on traditional subword segmentation
  methods.”, this is not definitively true. See discussion in [2].
  ○ Citation 25 is a LinkedIn post, which I don’t think represents rigorous enough
  scientific evidence to support the claim in the third to last paragraph on page 6.
  ○ The authors should also better introduce and summarize related work. For
  example, in “As shown in [23], even state-of-the-art LLMs struggle with
  compositional morphology”, the authors should explain what the paper tested
  before summarizing the findings. Otherwise, you can only follow paragraphs like
  this by reading the paper cited here.
  ● Repetition:
  ○ The point about 2.5 times token length is repeated on pages 5 and 6 in nearly
  identical contexts.
  ○ The second-to-last paragraph on p. 9 is repetitive with respect to UNK tokens.
  ● Important terms are not defined:
  ○ On p. 3 when you introduce TR% and Pure%, you need to define them. Similarly,
  if you’re going to use these terms in the abstract, they need to be introduced and
  defined, if only briefly.
  ○ Token purity is not defined at any point in the paper.○ In “Their analysis found that performance declines sharply as morphological
  complexity increases”, what is morphological complexity (within/across
  languages)? What is performance?
  ○ Compound words are introduced on p. 12, which have not been mentioned or
  defined up to that point. Do you treat them as root-root? How do they fit into the
  rest of the work?
  ○ Insufficient and possibly misleading explanations of final devoicing/haplology, and
  vowel hiatus. Final devoicing implies that the underlying form is voiced and when
  the form occurs without a vowel it is de-voiced, which is the opposite direction to
  what is shown.
  ● Structure and organization:
  ○ In the methodology section, the authors make claims about the performance of
  the tokenizer, but at that point in the paper, the results have not been introduced.
  Methodologically, I highlight some concerns here. In general, there is insufficient empirical
  evidence to support the authors’ claims and the evaluation methodology is poorly motivated.
  ● Token granularity (fertility) shouldn’t be compared across languages. Turkish words are
  longer, therefore, it is not surprising that there would be more tokens given a fixed vocab.
  Instead, the authors should use the method from [4] or similar.
  ● The authors also mention that this approach might improve summarization quality, but
  then don’t check that? Also, NER and sentiment analysis. Not mentioned is that some of
  these models were small bidirectional models. Do we expect all of those results to
  generalize to (larger) autoregressive models?
  ● I think if you’re going to make an argument about linguistic representations, you should
  evaluate on TurBLiMP [5] (or at least MultiBLiMP [6] which was released at the time of
  submission). In order to make this claim, the authors need to evaluate models on this
  benchmark, not tokenizers.
  ● “The results presented in this section provide strong empirical support”. There is no
  empirical analysis. The authors provide only a qualitative analysis of a small number of
  examples. This analysis is vaguely described and does not sufficiently support the
  claims.
  ● “The proposed framework provides a balance between linguistic integrity and
  computational efficiency.” The authors do not evaluate either of these.
  Clarification questions:
  ● Is the proposed tokenization method lossless?
  ● What is the internal composition of ”akarsu” and ” ̧cama ̧sırhane”. What do these words
  mean?
  ○ These examples use closing quotation marks at the beginning. Also, the other
  Turkish words are italicized, and these are inequities. You should use consistent
  formatting for clarity.
  ● The SentencePiece library supports Unigram and BPE tokenization. Which did you use?● What does “incorporated into the tokenizer” mean. Is this vocabulary expansion.
  ● Does this multi-step process slow things down?
  [1] https://aclanthology.org/2025.coling-main.441/
  [2] https://openreview.net/forum?id=XYRri1s6pP&noteId=XYRri1s6pP
  [3] https://aclanthology.org/2024.sigmorphon-1.4/
  [4]
  https://proceedings.neurips.cc/paper_files/paper/2023/file/74bb24dca8334adce292883b4b651e
  da-Paper-Conference.pdf
  [5] https://arxiv.org/pdf/2506.13487
  [6] https://arxiv.org/pdf/2504.02768?

- Reviewer from another journal:
  Strengths
  Technical novelty and innovationIntegrates dictionary-based morphological segmentation with BPE while explicitly normalizing Turkish phonological alternations (e.g., -dAn/-tAn, final devoicing) under shared IDs.
  Introduces practical engineering choices (e.g., case encoding via an token, explicit tokens for whitespace/newlines) to reduce vocabulary duplication without losing orthographic information.
  Provides both Python and Rust implementations with a clear, staged tokenization pipeline (root lookup, affix iteration, BPE fallback).
  Experimental rigor and validationReports TR% and Pure% on a sizable, Turkish-specific evaluation corpus (TR-MMLU), showing substantial gains over widely used tokenizers.
  Presents detailed qualitative analyses on complex Turkish sentences that showcase morpheme-aligned segmentations.
  Clarity of presentationThe motivation for morphology-aware tokenization is clearly articulated, with illustrative examples.
  The overall pipeline and decisions (root detection, affix inventory, phonological normalization, fallback strategy) are described in a reasonably transparent way.
  Significance of contributionsAddresses a well-known bottleneck for agglutinative languages: semantically incoherent and redundant subword segmentations produced by frequency-only tokenizers.
  If validated in downstream tasks, such tokenizers can have meaningful impact on Turkish NLP and potentially generalize to similar languages.

Weaknesses
Technical limitations or concernsThe method depends on curated root and affix dictionaries plus heuristic phonological normalization; the coverage, ambiguity handling, and error rates are not quantified.
Mapping multiple allomorphs to a shared ID risks information loss; the paper does not evaluate detokenization accuracy or ambiguities introduced by reverse mapping under context.
The uppercase-token mechanism is underspecified for edge cases (acronyms, mid-word caps, proper nouns with apostrophes in Turkish orthography).
Experimental gaps or methodological issuesNo downstream evaluation (LM perplexity, NLU/NLG tasks, or machine translation) to substantiate claims that higher TR%/Pure% translate into model performance gains.
The main metrics (TR% and Pure%) are computed over unique tokens and heavily favor morph-aware tokenizers by design; frequency-weighted variants, MorphScore, µ-consistency/µ-edit-distance, compression (CTC), Rényi entropy, and fertility are missing.
Baseline comparisons are potentially unfair: pretrained general-purpose tokenizers not optimized for Turkish are compared on linguistic-alignment metrics tailored for Turkish, with no vocab-size control or matched training regimen.
Lacks ablations isolating the effects of each component (phonological normalization, allomorph ID sharing, uppercase token, BPE size).
Clarity or presentation issuesSome flowchart artifacts and formatting issues from PDF extraction; minor but distracting.
The decoding process is described conceptually but without an algorithmic specification or measurable error analysis.
Missing related work or comparisonsRecent morphology-aware BPE approaches (e.g., MorphBPE; MorphTok/CBPE) and tokenizer-free/dynamic-chunking methods (e.g., H-NET++) are not empirically compared; their intrinsic metrics (MorphScore, morphological consistency and edit-distance) are not reported.
Unigram tokenizers, which often outperform BPE in morphologically rich settings, are not included in the comparisons.

Detailed Comments
Technical soundness evaluationThe hybrid design is sensible: a longest-match morphological pre-segmentation plus BPE fallback is a well-motivated and practical compromise. Phonological normalization for Turkish allomorphy is a valuable implementation detail.
However, several aspects require quantitative validation: (i) segmentation accuracy against a gold morphological segmentation; (ii) detokenization correctness (surface-form reconstruction rate); (iii) robustness to OOV roots/affixes and domain drift; (iv) ambiguity resolution when multiple roots/segmentations are plausible.
The decision to collapse allomorphs into shared IDs is linguistically motivated but risky for generation: without careful conditioning, the reverse mapping may produce surface errors, especially in noisy or low-context scenarios.
Experimental evaluation assessmentThe reported TR% and Pure% are strong, but they are intrinsic metrics that prioritize linguistic purity; alone, they do not establish efficacy for training or inference. The literature increasingly emphasizes multi-faceted intrinsic metrics (e.g., MorphScore, µ-consistency/µ-edit-distance) and downstream evaluations (perplexity, task scores).
Fertility/sequence-length implications are lightly addressed; higher interpretability can come with longer sequences and potentially higher compute. Reporting fertility, throughput, and memory would clarify trade-offs.
Fairness: Comparing a bespoke Turkish tokenizer to multilingual model tokenizers on Turkish-centric purity metrics is informative but cannot establish superiority for modeling effectiveness without matched training or at least downstream proxy tasks.
Comparison with related work (using the summaries provided)MorphPiece (2307.07262) demonstrates hybrid morphology+BPE for English with consistent downstream gains despite increased fertility; this paper should mirror that by training a small LM on Turkish with matched budgets to report perplexity and task performance.
MorphBPE (2502.00894) constrains merges to respect morpheme boundaries and introduces µ-consistency and µ-edit-distance; adding these metrics would position the proposed tokenizer in a broader methodological landscape and facilitate apples-to-apples comparison.
MorphTok/CBPE (2504.10335) offers script-sensitive constraints and morphology-aware pretokenization with empirical MT and LM wins; an Indic-to-Turkish transfer comparison is not expected but the evaluation protocol (fertility, human EvalTok, downstream) is instructive for this work.
Toraman et al. (2204.08832) show Turkish morphological tokenization is competitive with subword models; this paper aligns with that thrust but lacks the controlled downstream tests and vocabulary-size sweeps that study revealed are critical.
Tokenizer-free/dynamic-chunking (H-NET++) highlights a different path for MRLs; a discussion of computational cost and training feasibility relative to a morphology-aware tokenizer would strengthen the positioning.
Recent large-scale analyses (2411.14198) suggest byte-premium/data quantity drive cross-linguistic gaps more than morphology per se; this tempers the paper’s broader claims and motivates including byte-aware token budgets and downstream training controlled for data volume.
Discussion of broader impact and significanceIf validated downstream, the approach can enhance interpretability and reduce redundancy for Turkish and similar languages, potentially improving fairness in multilingual systems.
Risks include overfitting to prescriptive analyses, limited portability if dictionaries are incomplete, and increased maintenance burden for language-specific engineering.
Releasing code, dictionaries, and trained tokenizers with clear licenses would benefit the community and enable rigorous replication and extension.

Questions for Authors
Can you report detokenization accuracy (exact-surface reconstruction rate) and ambiguity/error analyses for reverse mapping, especially under allomorph ID sharing and in low-context settings?
What is the fertility (tokens per word) of your tokenizer on TR-MMLU versus baselines, and how does this impact throughput and memory in training/inference?
Could you add ablations isolating the contributions of: (a) phonological normalization, (b) allomorph-to-shared-ID mapping, (c) token, (d) BPE vocabulary size?
Do you have intrinsic morphological alignment scores against a gold or high-precision Turkish segmentation resource (e.g., MorphScore or boundary F1), and can you report µ-consistency/µ-edit-distance as in MorphBPE?
Can you provide downstream evidence (e.g., train a small decoder-only or encoder-only model on matched data/token budgets) showing perplexity or task improvements relative to BPE/Unigram and morphology-aware baselines?
How do you handle proper-noun capitalization with apostrophes (common in Turkish), acronyms, and mid-word capitalization? Are there failure cases with the token?
What is the coverage of your 22k-root and 230-affix dictionaries across domains, and how does performance degrade on out-of-domain or noisy text (e.g., social media, OCR)?
Will you release the tokenizer (code, dictionaries, BPE merges, evaluation scripts) to facilitate reproducibility and community comparison?

Overall Assessment
This paper tackles an important problem—morphology-aware tokenization for agglutinative languages—and proposes a practical hybrid approach with thoughtful Turkish-specific engineering. The intrinsic linguistic-alignment results (TR%/Pure%) and qualitative examples are promising and clearly demonstrate improved morpheme coherence compared to general-purpose tokenizers. However, the work falls short of NeurIPS standards in its current form due to limited experimental depth: no downstream modeling results, incomplete intrinsic diagnostics (fertility, MorphScore, µ-consistency/µ-edit-distance), lack of ablations, and fairness concerns in baseline selection. The claim of language independence is also untested beyond Turkish. Strengthening the paper with controlled downstream experiments (even at small scale), richer intrinsic metrics, ablations, and comparisons to morphology-aware BPE/Unigram baselines would significantly improve its rigor and impact. As it stands, the contribution is valuable but closer to a strong workshop paper; with the suggested additions, it could become a compelling submission for a top-tier venue.
