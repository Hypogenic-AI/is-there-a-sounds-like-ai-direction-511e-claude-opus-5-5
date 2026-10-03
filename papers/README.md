# Downloaded Papers (29)

Paper-finder service was down (HTTP 500), so papers were found via arXiv API keyword searches (logs/arxiv_search*.txt). `paper_meta.json` holds full metadata + abstracts. Detailed notes for deep-read papers are in `papers/notes/`.

1. [Linear Probing Provides Robust and Efficient Detection of Machine-Generated Text](2608.24780_linear_probing_provides_robust_and_efficient_detec.pdf)
   - Authors: Gerrit Quaremba, Hanqi Yan, Elizabeth Black, Denny Vrandecic et al.
   - Year: 2026; arXiv: 2608.24780
   - Why relevant: [USER] Linear probes on Llama-3-8B last-token states detect MGT; shared 'machineness' direction; no causal test. Deep-read: notes/2608.24780_notes.md

2. [SV-Detect: AI-generated Text Detection with Steering Vectors](2606.07313_sv_detect_ai_generated_text_detection_with_steerin.pdf)
   - Authors: Mikhail Vishnyakov, Tatiana Gaintseva
   - Year: 2026; arXiv: 2606.07313
   - Why relevant: [USER] Per-layer human-vs-AI directions as detector features; logit-lens shows formal-vs-casual axis; never steers. Deep-read: notes/2606.07313_notes.md

3. [Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders](2503.03601_feature_level_insights_into_artificial_text_detect.pdf)
   - Authors: Kristian Kuznetsov, Laida Kushnareva, Polina Druzhinina, Anton Razzhigaev et al.
   - Year: 2025; arXiv: 2503.03601
   - Why relevant: [USER] Gemma Scope SAE features for AI-text detection (Gemma-2-2b L16); qualitative steering only. Deep-read: notes/2503.03601_notes.md

4. [The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models](2601.10387_the_assistant_axis_situating_and_stabilizing_the_d.pdf)
   - Authors: Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish et al.
   - Year: 2026; arXiv: 2601.10387
   - Why relevant: [USER] Assistant Axis = default-assistant minus role-play mean difference; key persona confound + steering recipe. Deep-read: notes/2601.10387_notes.md

5. [Steer-to-Detect: Probing Hidden Representations for Detection of LLM-Generated Texts](2605.12890_steer_to_detect_probing_hidden_representations_for.pdf)
   - Authors: Luxu Liang, Xiang Li
   - Year: 2026; arXiv: 2605.12890
   - Why relevant: Learned steering vector applied while reading improves detection; formality-heavy token effects. Deep-read: notes/2605.12890_notes.md

6. [RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors](2405.07940_raid_a_shared_benchmark_for_robust_evaluation_of_m.pdf)
   - Authors: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan et al.
   - Year: 2024; arXiv: 2405.07940
   - Why relevant: RAID benchmark (dataset we use); detector robustness across domains/attacks/decoding

7. [MAGE: Machine-generated Text Detection in the Wild](2305.13242_mage_machine_generated_text_detection_in_the_wild.pdf)
   - Authors: Yafu Li, Qintong Li, Leyang Cui, Wei Bi et al.
   - Year: 2023; arXiv: 2305.13242
   - Why relevant: MAGE benchmark (dataset we use)

8. [How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection](2301.07597_how_close_is_chatgpt_to_human_experts_comparison_c.pdf)
   - Authors: Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang et al.
   - Year: 2023; arXiv: 2301.07597
   - Why relevant: HC3 corpus (dataset we use); human vs ChatGPT linguistic differences

9. [Do LLMs write like humans? Variation in grammatical and rhetorical styles](2410.16107_do_llms_write_like_humans_variation_in_grammatical.pdf)
   - Authors: Alex Reinhart, Ben Markey, Michael Laudenbach, Kachatad Pantusen et al.
   - Year: 2024; arXiv: 2410.16107
   - Why relevant: HAP-E parallel corpus; instruct-tuned models deviate stylistically, base models ≈ human (notes in 2605.12890_notes.md)

10. [Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text](2401.12070_spotting_llms_with_binoculars_zero_shot_detection.pdf)
   - Authors: Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi et al.
   - Year: 2024; arXiv: 2401.12070
   - Why relevant: Binoculars zero-shot detector (independent evaluator)

11. [Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature](2310.05130_fast_detectgpt_efficient_zero_shot_detection_of_ma.pdf)
   - Authors: Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang et al.
   - Year: 2023; arXiv: 2310.05130
   - Why relevant: Fast-DetectGPT zero-shot detector (independent evaluator)

12. [DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature](2301.11305_detectgpt_zero_shot_machine_generated_text_detecti.pdf)
   - Authors: Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning et al.
   - Year: 2023; arXiv: 2301.11305
   - Why relevant: DetectGPT, foundational zero-shot detector

13. [Idiosyncrasies in Large Language Models](2502.12150_idiosyncrasies_in_large_language_models.pdf)
   - Authors: Mingjie Sun, Yida Yin, Zhiqiu Xu, J. Zico Kolter et al.
   - Year: 2025; arXiv: 2502.12150
   - Why relevant: LLM idiosyncrasies are word-level & survive rewriting — lexical confound

14. [Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors](2505.24523_stress_testing_machine_generated_text_detection_sh.pdf)
   - Authors: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi et al.
   - Year: 2025; arXiv: 2505.24523
   - Why relevant: DPO to shift LLM style toward human fools detectors (fine-tuning analogue of our steering) (notes in 2605.12890_notes.md)

15. [AuthorMist: Evading AI Text Detectors with Reinforcement Learning](2503.08716_authormist_evading_ai_text_detectors_with_reinforc.pdf)
   - Authors: Isaac David, Arthur Gervais
   - Year: 2025; arXiv: 2503.08716
   - Why relevant: RL humanizer evading detectors; baseline for 'less AI' generation

16. [Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts](2306.04723_intrinsic_dimension_estimation_for_robust_detectio.pdf)
   - Authors: Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii et al.
   - Year: 2023; arXiv: 2306.04723
   - Why relevant: Intrinsic dimension of human vs AI text embeddings; geometric difference

17. [Steering Language Models With Activation Engineering](2308.10248_steering_language_models_with_activation_engineeri.pdf)
   - Authors: Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell et al.
   - Year: 2023; arXiv: 2308.10248
   - Why relevant: ActAdd: steering by adding activation differences

18. [Steering Llama 2 via Contrastive Activation Addition](2312.06681_steering_llama_2_via_contrastive_activation_additi.pdf)
   - Authors: Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong et al.
   - Year: 2023; arXiv: 2312.06681
   - Why relevant: CAA: mean-difference steering vectors; standard recipe

19. [Representation Engineering: A Top-Down Approach to AI Transparency](2310.01405_representation_engineering_a_top_down_approach_to.pdf)
   - Authors: Andy Zou, Long Phan, Sarah Chen, James Campbell et al.
   - Year: 2023; arXiv: 2310.01405
   - Why relevant: Representation Engineering: reading & controlling concepts

20. [Refusal in Language Models Is Mediated by a Single Direction](2406.11717_refusal_in_language_models_is_mediated_by_a_single.pdf)
   - Authors: Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka et al.
   - Year: 2024; arXiv: 2406.11717
   - Why relevant: Refusal direction: directional ablation/addition methodology (weight orthogonalization)

21. [Persona Vectors: Monitoring and Controlling Character Traits in Language Models](2507.21509_persona_vectors_monitoring_and_controlling_charact.pdf)
   - Authors: Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans et al.
   - Year: 2025; arXiv: 2507.21509
   - Why relevant: Persona vectors: automated trait-vector pipeline, projection monitoring

22. [The Linear Representation Hypothesis and the Geometry of Large Language Models](2311.03658_the_linear_representation_hypothesis_and_the_geome.pdf)
   - Authors: Kiho Park, Yo Joong Choe, Victor Veitch
   - Year: 2023; arXiv: 2311.03658
   - Why relevant: Linear representation hypothesis: theory, causal inner product

23. [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](2310.06824_the_geometry_of_truth_emergent_linear_structure_in.pdf)
   - Authors: Samuel Marks, Max Tegmark
   - Year: 2023; arXiv: 2310.06824
   - Why relevant: Geometry of truth: probe vs mean-difference directions, causal tests

24. [A Unified Understanding and Evaluation of Steering Methods](2502.02716_a_unified_understanding_and_evaluation_of_steering.pdf)
   - Authors: Shawn Im, Sharon Li
   - Year: 2025; arXiv: 2502.02716
   - Why relevant: Unified evaluation of steering methods (mean-diff favoured)

25. [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](2501.17148_axbench_steering_llms_even_simple_baselines_outper.pdf)
   - Authors: Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang et al.
   - Year: 2025; arXiv: 2501.17148
   - Why relevant: AxBench: steering eval protocol (concept/instruction/fluency judge scores); DiffMean strong

26. [On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study](2606.12234_on_the_effectiveness_fluency_trade_off_in_llm_cond.pdf)
   - Authors: Iuri Macocco, Pau Rodríguez, Arno Blaas, Luca Zappella et al.
   - Year: 2026; arXiv: 2606.12234
   - Why relevant: Effectiveness-fluency trade-off; steering weaker on instruct than base models

27. [Style Vectors for Steering Generative Large Language Model](2402.01618_style_vectors_for_steering_generative_large_langua.pdf)
   - Authors: Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt et al.
   - Year: 2024; arXiv: 2402.01618
   - Why relevant: Style vectors from activations steer sentiment/emotion/style

28. [On the Limits of Steering Vectors for Preference-Aligned Generation](2607.01802_on_the_limits_of_steering_vectors_for_preference_a.pdf)
   - Authors: Melanie Subbiah, Zara Hall, Kathleen McKeown
   - Year: 2026; arXiv: 2607.01802
   - Why relevant: Limits of steering for writing-style preferences on Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct

29. [LEACE: Perfect linear concept erasure in closed form](2306.03819_leace_perfect_linear_concept_erasure_in_closed_for.pdf)
   - Authors: Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell et al.
   - Year: 2023; arXiv: 2306.03819
   - Why relevant: LEACE concept erasure — remove direction / confound-controlled erasure
