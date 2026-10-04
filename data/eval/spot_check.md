# Eval-set spot check (20 questions)

For each item: is the question sensible, and does the quoted evidence actually answer it? For unanswerable items: confirm the answer is NOT in the documents.


## 1. `ai-22`  (fact, mode=any, split=dev)

**Q:** After completing the MAP function, what initial decision should users be able to make?

- **Gold quote** (nist_ai_rmf_1_0): `to inform an initial go/no-go decision about whether to design, develop, or deploy an AI system`
  - found in `nist_ai_rmf_1_0:p30:c1` (page 30)


## 2. `lim-10`  (fact, mode=any, split=dev)

**Q:** At which Llama-2 sizes does the U-shaped position curve show up?

- **Gold quote** (lost_in_the_middle_2023): `the 7B Llama-2 models are solely recency biased, while the 13B and 70B models exhibit a U-shaped performance curve`
  - found in `lost_in_the_middle_2023:p9:c1` (page 9)


## 3. `csf-08`  (fact, mode=any, split=dev)

**Q:** What does the DETECT function enable?

- **Gold quote** (nist_csf_2_0): `DETECT enables the timely discovery and analysis of anomalies, indicators of compromise, and other potentially adverse events`
  - found in `nist_csf_2_0:p9:c0` (page 9)


## 4. `pico-20`  (fact, mode=any, split=test)

**Q:** What resolution and sample rate does the analogue-to-digital converter have?

- **Gold quote** (raspberry_pi_pico_datasheet): `12-bit 500 ksps Analogue to Digital Converter (ADC)`
  - found in `raspberry_pi_pico_datasheet:p5:c0` (page 5)


## 5. `bl-06`  (fact, mode=any, split=test)

**Q:** How does Berkshire's net worth compare with the total for the other 499 S&P companies?

- **Gold quote** (berkshire_2023_excerpt): `The total GAAP net worth for the other 499 S&P companies – a who’s who of American business – was $8.9 trillion in 2022`
  - found in `berkshire_2023_excerpt:p4:c0` (page 4)


## 6. `bh-05`  (table, mode=any, split=dev)

**Q:** How much did Berkshire spend acquiring businesses, net of cash acquired, in 2023?

- **Gold quote** (berkshire_2023_excerpt): `Acquisitions of businesses, net of cash acquired|(8,604`
  - found in `berkshire_2023_excerpt:p29:c2` (page 29)


## 7. `bh-07`  (table, mode=any, split=dev)

**Q:** What were GEICO's revenues and pre-tax earnings in 2023?

- **Gold quote** (berkshire_2023_excerpt): `GEICO|$ 39,264|$ 38,984|$ 37,706|$ 3,635`
  - found in `berkshire_2023_excerpt:p31:c1` (page 31)


## 8. `rag-05`  (table, mode=any, split=test)

**Q:** What Natural Questions exact-match score did RAG-Token reach in the ablation that swapped in a fixed BM25 retriever?

- **Gold quote** (rag_lewis_2020): `RAG-Token-BM25|29.7|41.5|32.1|33.1`
  - found in `rag_lewis_2020:p8:c1` (page 8)


## 9. `bh-04`  (table, mode=any, split=test)

**Q:** What were net earnings per average equivalent Class A share in 2023?

- **Gold quote** (berkshire_2023_excerpt): `Net earnings (loss) per average equivalent Class A share|$ 66,412`
  - found in `berkshire_2023_excerpt:p27:c3` (page 27)


## 10. `bh-15`  (table, mode=any, split=dev)

**Q:** How much cash did Berkshire spend buying back its own shares in 2023 according to the cash flow statement?

- **Gold quote** (berkshire_2023_excerpt): `Acquisitions of treasury stock|(9,171`
  - found in `berkshire_2023_excerpt:p29:c3` (page 29)


## 11. `bl-07`  (paraphrase, mode=any, split=test)

**Q:** What is the one investment rule at Berkshire that will never change?

- **Gold quote** (berkshire_2023_excerpt): `One investment rule at Berkshire has not and will not change: Never risk permanent loss of capital`
  - found in `berkshire_2023_excerpt:p5:c1` (page 5)


## 12. `lim-05`  (paraphrase, mode=any, split=dev)

**Q:** Why does reader accuracy stop improving even though the retriever keeps finding more relevant documents as k grows?

- **Gold quote** (lost_in_the_middle_2023): `reader model performance saturates long before retriever performance saturates, indicating that readers are not effectively using the extra context`
  - found in `lost_in_the_middle_2023:p10:c0` (page 10)


## 13. `csf-07`  (paraphrase, mode=any, split=test)

**Q:** Which function covers containing the effects of a cybersecurity incident, including analysis, mitigation and reporting?

- **Gold quote** (nist_csf_2_0): `RESPOND supports the ability to contain the effects of cybersecurity incidents`
  - found in `nist_csf_2_0:p9:c0` (page 9)


## 14. `csf-06`  (paraphrase, mode=any, split=test)

**Q:** What is the step after creating an Organizational Profile that identifies differences between where you are and where you want to be?

- **Gold quote** (nist_csf_2_0): `Analyze the gaps between the Current and Target Profiles, and create an action plan`
  - found in `nist_csf_2_0:p12:c0` (page 12)


## 15. `lim-03`  (paraphrase, mode=any, split=test)

**Q:** Which retriever was used to pick the hard-negative distractor passages for the long-context position experiments?

- **Gold quote** (lost_in_the_middle_2023): `we use a retrieval system (Contriever, fine-tuned on MS-MARCO; Izacard et al., 2021) to retrieve the`
  - found in `lost_in_the_middle_2023:p3:c1` (page 3)


## 16. `csf-04`  (multi, mode=all, split=dev)

**Q:** What is the difference between a Current Profile and a Target Profile?

- **Gold quote** (nist_csf_2_0): `A Current Profile specifies the Core outcomes that an organization is currently achieving`
  - found in `nist_csf_2_0:p11:c0` (page 11)
- **Gold quote** (nist_csf_2_0): `A Target Profile specifies the desired outcomes that an organization has selected and prioritized`
  - found in `nist_csf_2_0:p11:c0` (page 11)


## 17. `lim-09`  (multi, mode=all, split=test)

**Q:** How did the best-to-worst position gap compare between MPT-30B and its instruction-tuned version, and what does that say about instruction tuning?

- **Gold quote** (lost_in_the_middle_2023): `instruction fine-tuning slightly reduces the worst-case performance disparity from nearly 10% between the base model best- and worst-case performance to around 4%`
  - found in `lost_in_the_middle_2023:p9:c0` (page 9)
  - found in `lost_in_the_middle_2023:p9:c1` (page 9)
- **Gold quote** (lost_in_the_middle_2023): `indicating that the instruction fine-tuning process itself is not necessarily responsible for these performance trends`
  - found in `lost_in_the_middle_2023:p9:c0` (page 9)


## 18. `rag-08`  (multi, mode=all, split=test)

**Q:** What is the difference between the RAG-Sequence and RAG-Token formulations?

- **Gold quote** (rag_lewis_2020): `In one approach, RAG-Sequence, the model uses the same document to predict each target token`
  - found in `rag_lewis_2020:p3:c0` (page 3)
- **Gold quote** (rag_lewis_2020): `The second approach, RAG-Token, can predict each target token based on a different document`
  - found in `rag_lewis_2020:p3:c0` (page 3)


## 19. `ai-15`  (unanswerable, mode=any, split=test)

**Q:** What financial penalty does the AI RMF impose on organizations that fail to follow it?

**Expected: unanswerable** (no gold evidence)


## 20. `bh-24`  (unanswerable, mode=any, split=test)

**Q:** What was Occidental Petroleum's closing share price on December 29, 2023?

**Expected: unanswerable** (no gold evidence)
