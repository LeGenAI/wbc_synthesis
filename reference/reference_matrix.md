# Reference Matrix for Research Audit

## 목적

이 문서는 WBC synthetic augmentation / WBC domain generalization / 일반 medical DG augmentation 관련 선행연구를 `로컬에 내려받은 원문 또는 공식 저장소` 기준으로만 정리한 reference matrix다.  
목표는 `hallucination 없이` 다음 두 질문에 답할 수 있도록 만드는 것이다.

1. 이미 존재하는 strong baseline은 무엇인가
2. 우리가 향후 제안할 generation policy는 어디에서만 novelty를 주장할 수 있는가

## 검증 규칙

1. 이 문서의 항목은 모두 `reference/sources/` 아래에 로컬 사본이 있는 경우에만 포함했다.
2. 각 요약은 원문 제목, 초록, DOI BibTeX, 공식 GitHub README에서 직접 확인 가능한 수준으로만 썼다.
3. source 간 제목/연도/venue 표기가 어긋나는 경우에는 `source note`로 따로 적었다.
4. 아직 원문/코드가 불충분한 항목은 넣지 않았다.

## 로컬 소스 구성

- 논문/landing page: `reference/sources/papers/`
- 공식 코드 README: `reference/sources/repos/`
- DOI/arXiv BibTeX: `reference/sources/bibtex/`
- 통합 BibTeX: `reference/references.bib`
- CSV 매트릭스: `reference/reference_matrix.csv`

## Matrix

| Key | Bucket | Verified source basis | What it actually does | Why it matters for us | Gap vs our intended claim |
| --- | --- | --- | --- | --- | --- |
| `almezhghwi2020` | WBC synthetic augmentation | PMC article + DOI BibTeX | GAN and image transforms for 5-class WBC classification on LISC; reports strong in-domain validation accuracy. | WBC synthesis가 새롭지 않다는 점을 분명히 보여준다. | cross-domain benchmark, leakage-safe evaluation, policy-level subset selection이 없다. |
| `jung2022` | WBC synthetic augmentation | BMC PDF metadata + DOI BibTeX | CNN classification and generative models for WBC images; synthetic data generation is part of classification pipeline. | WBC synthetic data release / generation-classification 결합의 기존 예시다. | unseen-domain utility가 아니라 in-domain classification improvement에 가깝다. |
| `tavakoli2021` | WBC generalizability | bioRxiv HTML/PDF + DOI BibTeX | Raabin-trained model을 다른 dataset에 적용하며 WBC generalizability gap을 문제로 제기한다. | 우리 LODO framing의 정당성을 외부에서 뒷받침한다. | synthetic generation이나 augmentation policy 자체는 제안하지 않는다. |
| `umer2023` | WBC domain generalization | arXiv abs/PDF/BibTeX | domain shift와 class imbalance를 동시에 다루는 hematological cytomorphology DG classifier를 제안한다. | WBC DG 자체도 이미 선행이 있다는 점을 보여준다. | classifier loss / representation 중심이며 synthetic pool design은 아니다. |
| `putzu2025` | WBC cross-domain inference | J. Imaging PDF + DOI BibTeX | OOD filtering과 self-ensembling을 결합한 test-time augmentation으로 cross-domain leukocyte classification을 개선한다. | cross-domain WBC 개선이 training-time synthesis 없이도 가능함을 보여주는 강한 비교축이다. | training-time generation policy나 synthetic subset utility를 다루지 않는다. |
| `boada2025` | WBC diffusion synthesis | ICCVW paper PDF + DOI BibTeX + official repo README | LoRA-finetuned diffusion model `CytoDiff`로 synthetic cytomorphology images를 만들고 low-data classifier gains를 보고한다. | diffusion-based WBC generation은 이미 존재한다는 점을 확인시킨다. | low-data/class imbalance 개선이 중심이며 leakage-safe multi-domain generalization 평가는 아니다. |
| `hassini2025` | WBC rare-class synthesis | Nature HTML + DOI BibTeX | physics-inspired GAN으로 FPM intensity/phase bimodal WBC data를 생성하고 rare basophil classification을 보강한다. | rare-class augmentation과 physics-aware generation이 이미 존재함을 보여준다. | modality가 FPM으로 특수하고, smear-image DG나 synthetic policy selection과는 다르다. |
| `li2024_sada` | WBC DG with augmentation/alignment | arXiv abs/PDF/BibTeX + official repo README | stain-based augmentation, domain alignment, supervised contrastive learning을 결합해 여러 WBC dataset에서 SOTA를 주장한다. | 현재 외부 WBC DG baseline 중 우리와 가장 가까운 경쟁축이다. | generation model이 아니라 classifier/feature learning 중심이고, synthetic pool utility benchmark는 없다. |
| `li2020_ldr` | General medical DG | arXiv abs/PDF/BibTeX | linear-dependency regularization으로 medical imaging classification의 cross-domain generalization을 개선한다. | 일반 medical DG에서 representation-level 해법이 이미 강하다는 점을 보여준다. | WBC 특화가 아니고 synthetic generation policy는 없다. |
| `su2022_slaug` | General medical DG augmentation | arXiv abs/PDF/BibTeX + official repo README | global/local augmentation과 saliency balancing으로 single-source medical DG segmentation을 개선한다. | `augmentation policy 자체`가 novelty가 될 수 있음을 보여주는 핵심 precedent다. | segmentation이며 WBC classification도, synthetic image generation도 아니다. |
| `shen2025_stycona` | General medical DG augmentation | arXiv abs/PDF/BibTeX + official repo README | style-content decomposition으로 style shift와 content shift를 분리해 DG segmentation augmentation을 설계한다. | 우리 contextual / cell-preservation 사고와 가장 닮은 외부 축이다. | segmentation task이고, WBC synthetic pool utility benchmark는 없다. |
| `doerrich2026_stylizingvit` | General medical DG augmentation | arXiv abs/PDF/BibTeX + official repo README | anatomy-preserving style transfer를 통한 DG classification/TTA를 제안한다. | style diversity를 늘리되 anatomy를 보존하는 framing이 이미 존재함을 보여준다. | WBC가 아니고, synthetic subset selection이나 leakage-safe LODO utility는 다루지 않는다. |

## Item-by-item review

### 1. `almezhghwi2020`

- Citation:
  Almezhghwi, Khaled and Sertan Serte. *Improved Classification of White Blood Cells with the Generative Adversarial Network and Deep Convolutional Neural Network*. Computational Intelligence and Neuroscience, 2020.
- Local sources:
  `reference/sources/papers/almezhghwi2020_pmc.html`
  `reference/sources/bibtex/almezhghwi2020.bib`
- Review:
  이 논문은 WBC 5분류에서 GAN 기반 synthetic augmentation과 일반 image transform augmentation을 함께 비교한다. 핵심 메시지는 `WBC synthetic augmentation이 classification을 도울 수 있다`는 것이지, domain shift 환경에서 어떤 synthetic policy가 utility를 높이는가가 아니다. 따라서 이 논문을 넘으려면 `WBC에 diffusion/LoRA를 쓴다`는 수준이 아니라, 더 어려운 평가 무대와 더 정교한 policy를 제시해야 한다.

### 2. `jung2022`

- Citation:
  Jung, Changhun, Mohammed Abuhamad, David Mohaisen, Kyungja Han, and DaeHun Nyang. *WBC image classification and generative models based on convolutional neural network*. BMC Medical Imaging, 2022.
- Local sources:
  `reference/sources/papers/jung2022_bmc.pdf`
  `reference/sources/bibtex/jung2022.bib`
- Review:
  이 논문은 WBC classification과 generative model을 같은 프레임 안에서 다루며, synthetic data release까지 언급되는 축이다. 즉, `WBC 합성 이미지를 만들어 classifier를 개선한다`는 내러티브는 이미 legacy가 있다. 다만 unseen-domain 일반화나 leakage-safe holdout 설계는 보이지 않으므로, 우리 쪽 차별점은 `target domain을 직접 보지 않는 utility benchmark` 쪽에서 더 안전하다.

### 3. `tavakoli2021`

- Citation:
  Tavakoli, Sajad, Ali Ghaffari, and Zahra Mousavi Kouzehkanan. *Generalizability in White Blood Cells’ Classification Problem*. bioRxiv, 2021.
- Local sources:
  `reference/sources/papers/tavakoli2021_biorxiv.html`
  `reference/sources/papers/tavakoli2021_biorxiv.pdf`
  `reference/sources/bibtex/tavakoli2021.bib`
- Review:
  이 논문은 WBC classification의 핵심 병목이 단순 accuracy가 아니라 dataset generalizability라는 점을 직접 제기한다. synthetic generation을 쓰지 않더라도, 우리 연구가 `왜 LODO와 held-out domain을 메인 benchmark로 잡아야 하는가`를 정당화하는 데 매우 유용하다. 반대로 말하면, 일반화 문제를 제기하는 것 자체는 novelty가 아니다.

### 4. `umer2023`

- Citation:
  Umer, Rao Muhammad, Armin Gruber, Sayedali Shetab Boushehri, Christian Metak, and Carsten Marr. *Imbalanced Domain Generalization for Robust Single Cell Classification in Hematological Cytomorphology*. arXiv, 2023.
- Local sources:
  `reference/sources/papers/2303.07771.html`
  `reference/sources/papers/umer2023_imbalanced_dg.pdf`
  `reference/sources/bibtex/2303.07771.bib`
- Review:
  이 논문은 WBC/hematological cytomorphology에서 domain shift와 class imbalance를 함께 다루는 classifier-centric DG 연구다. 따라서 `WBC에서 DG를 한다`는 주장도 이미 선행이 있다. 우리 축이 다르려면 classifier objective가 아니라 `synthetic generation policy`와 `policy evaluation protocol`에서 차별화되어야 한다.

### 5. `putzu2025`

- Citation:
  Putzu, Lorenzo, Andrea Loddo, and Cecilia Di Ruberto. *Test-Time Augmentation for Cross-Domain Leukocyte Classification via OOD Filtering and Self-Ensembling*. Journal of Imaging, 2025.
- Local sources:
  `reference/sources/papers/putzu2025_jimaging.pdf`
  `reference/sources/bibtex/putzu2025.bib`
- Review:
  이 논문은 cross-domain leukocyte classification을 개선하는 방식으로 test-time augmentation과 OOD filtering을 사용한다. training-time synthetic generation이 없어도 cross-domain 성능 개선을 낼 수 있다는 점에서, 우리 방법은 반드시 `왜 generation policy가 필요한가`를 설명해야 한다. 즉, 본 연구와의 비교는 `synthetic policy가 inference-time adaptation과 다른 장점을 주는가`라는 질문으로 정리되어야 한다.

### 6. `boada2025`

- Citation:
  Boada, Jan Carreras, Rao Muhammad Umer, and Carsten Marr. *CytoDiff: AI-Driven Cytomorphology Image Synthesis for Medical Diagnostics*. ICCV Workshops, 2025.
- Local sources:
  `reference/sources/papers/boada2025_cytodiff_iccvw.pdf`
  `reference/sources/repos/cytodiff_README.md`
  `reference/sources/bibtex/boada2025.bib`
- Review:
  CytoDiff는 diffusion + LoRA 기반 cytomorphology image synthesis를 명시적으로 다루며, low-data / imbalanced classification gain을 보고한다. 따라서 `diffusion으로 WBC 계열 이미지를 생성하고 downstream classifier를 올린다`는 큰 메시지는 이미 존재한다. 우리 쪽이 주장할 수 있는 차이는 multi-domain held-out utility, leakage-safe filtering, 그리고 생성량이 아니라 policy selection에 있다.

### 7. `hassini2025`

- Citation:
  Hassini, Houda, Bernadette Dorizzi, Vincent Leymarie, Jacques Klossa, and Yaneck Gottesman. *Enhancing classification of rare white blood cells in FPM with a physics-inspired GAN*. Scientific Reports, 2025.
- Local sources:
  `reference/sources/papers/hassini2025_nature.html`
  `reference/sources/bibtex/hassini2025.bib`
- Review:
  이 논문은 rare basophil classification을 위해 physics-inspired GAN을 사용하며, intensity/phase coupling이 있는 FPM modality를 전제로 한다. 희귀 클래스 보강과 physics-aware generation이라는 명확한 contribution이 있으므로, 우리 쪽이 `rare class augmentation`만으로 novelty를 주장하기는 어렵다. 반면 우리는 일반 blood smear multi-domain setting과 utility-aware policy benchmark로 분리할 수 있다.

### 8. `li2024_sada`

- Citation:
  Li, Yongcheng, Lingcong Cai, Ying Lu, Xianghua Fu, Xiao Han, Ma Li, Wenxing Lai, Xiangzhong Zhang, and Xiaomao Fan. *Stain-aware Domain Alignment for Imbalance Blood Cell Classification*. arXiv, 2024.
- Local sources:
  `reference/sources/papers/2412.02976.html`
  `reference/sources/papers/li2024_sada_arxiv.pdf`
  `reference/sources/repos/sada_README.md`
  `reference/sources/bibtex/2412.02976.bib`
- Review:
  SADA는 stain-aware augmentation, local alignment, supervised contrastive learning을 결합한 강한 WBC DG baseline이다. 네 개의 public dataset과 private dataset을 함께 사용한다는 점도 우리와 매우 가깝다. 다만 이 축은 feature learning / domain alignment가 중심이며 synthetic image generation policy를 직접 설계하는 논문은 아니다.
- Source note:
  arXiv 제목은 `Imbalance Blood Cell Classification`이고, repo subtitle은 `Imbalanced Blood Cell Classification`이다. 매트릭스에서는 arXiv 표기를 우선했다.

### 9. `li2020_ldr`

- Citation:
  Li, Haoliang, YuFei Wang, Renjie Wan, Shiqi Wang, Tie-Qiang Li, and Alex C. Kot. *Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization*. arXiv, 2020.
- Local sources:
  `reference/sources/papers/2009.12829.html`
  `reference/sources/papers/li2020_ldr.pdf`
  `reference/sources/bibtex/2009.12829.bib`
- Review:
  이 논문은 representation regularization으로 unseen medical domains에 generalize하는 classifier를 설계한다. 즉, 일반 medical DG에서는 synthetic generation이 아니라 feature-space regularization이 이미 강한 baseline이라는 뜻이다. 따라서 우리 논문이 설득력을 가지려면 단순 accuracy 비교가 아니라 `generation policy가 왜 필요한가`를 더 분명히 해야 한다.

### 10. `su2022_slaug`

- Citation:
  Su, Zixian, Kai Yao, Xi Yang, Qiufeng Wang, Jie Sun, and Kaizhu Huang. *Rethinking Data Augmentation for Single-source Domain Generalization in Medical Image Segmentation*. arXiv, 2022.
- Local sources:
  `reference/sources/papers/2211.14805.html`
  `reference/sources/papers/su2023_slaug_arxiv.pdf`
  `reference/sources/repos/slaug_README.md`
  `reference/sources/bibtex/su2023_slaug.bib`
- Review:
  SLAug는 medical DG에서 `augmentation policy 자체를 다시 설계`하는 것이 contribution이 될 수 있음을 보여준다. global-only augmentation이 부족하다는 문제의식, local/global 제어, saliency balancing 등은 우리 generation policy framing과 직접 연결된다. 다만 segmentation이고 WBC가 아니라는 점에서, 그대로 가져오기는 어렵고 `policy-level novelty의 precedent`로 보는 것이 맞다.
- Source note:
  arXiv preprint 연도는 2022이고, 공식 repo는 AAAI 2023 implementation으로 표기한다.

### 11. `shen2025_stycona`

- Citation:
  Shen, Zhiqiang, Peng Cao, Jinzhu Yang, Osmar R. Zaiane, and Zhaolin Chen. *Style Content Decomposition-based Data Augmentation for Domain Generalizable Medical Image Segmentation*. arXiv, 2025.
- Local sources:
  `reference/sources/papers/2502.20619.html`
  `reference/sources/papers/shen2025_stycona_arxiv.pdf`
  `reference/sources/repos/stycona_README.md`
  `reference/sources/bibtex/2502.20619.bib`
- Review:
  StyCona는 style shift와 content shift를 분리해서 augmentation policy를 설계한다. 이는 우리 contextual preprocessing, cell-preservation, background shift 아이디어와 가장 가까운 외부 reference다. 다만 task가 segmentation이므로, 우리 쪽 novelty는 `세포 형태 보존 + domain-shift utility`를 classification synthetic pool 관점에서 다시 정의하는 데 있어야 한다.
- Source note:
  arXiv는 2025 preprint이고, 공식 repo citation은 MIDL 2026으로 적혀 있다.

### 12. `doerrich2026_stylizingvit`

- Citation:
  Doerrich, Sebastian, Francesco Di Salvo, Jonas Alle, and Christian Ledig. *Stylizing ViT: Anatomy-Preserving Instance Style Transfer for Domain Generalization*. arXiv, 2026.
- Local sources:
  `reference/sources/papers/2601.17586.html`
  `reference/sources/papers/doerrich2026_stylizingvit_arxiv.pdf`
  `reference/sources/repos/stylizingvit_README.md`
  `reference/sources/bibtex/2601.17586.bib`
- Review:
  Stylizing ViT는 anatomy-preserving style transfer를 DG classification과 TTA에 동시에 쓰는 방법이다. `style diversity를 늘리되 anatomy는 깨지지 않아야 한다`는 메시지는 우리 generation policy 설계에서 이미 선행이 있다. 따라서 우리가 주장할 수 있는 novelty는 style preservation 자체가 아니라, 이를 WBC synthetic utility benchmark와 연결하는 방식에 더 가깝다.
- Source note:
  arXiv preprint는 2026이며, 공식 repo는 ISBI 2026 표기를 사용한다.

## 감사 단계의 중간 결론

### 1. 이미 존재하는 것

- WBC synthetic augmentation 자체
- WBC cross-dataset generalizability 문제 제기
- WBC domain generalization classifier
- medical DG에서 augmentation policy redesign
- anatomy/style-content preservation 기반 DG augmentation

### 2. 아직 비어 있는 조합

- `WBC synthetic generation policy`를 `leakage-safe multi-domain utility benchmark` 위에서 직접 비교하는 프레임
- 생성량이나 단순 fidelity보다 `downstream utility`를 중심으로 synthetic pool을 평가하는 프레임
- `cell identity preservation`과 `domain style shift`를 동시에 조절하면서, 그 효과를 held-out domain F1로 닫는 프레임

### 3. RQ로 넘길 수 있는 가장 안전한 wedge

- 약한 표현:
  `WBC domain shift 환경에서 synthetic augmentation의 효과는 generation quantity보다 policy design에 더 크게 좌우된다`
- 그보다 강한 표현:
  `To our best knowledge, prior WBC synthetic studies는 주로 in-domain / low-data / class-imbalance gains에 머물렀고, prior WBC DG studies는 주로 classifier or adaptation methods에 초점을 두었다. 따라서 leakage-safe LODO benchmark 위에서 generation policy 자체를 비교하는 framing은 별도로 정리될 여지가 있다.`

### 4. 아직 조심해야 하는 주장

- `우리가 최초다`
- `기존 방법을 전반적으로 능가한다`
- `boundary-aware/contextual generation이 완전히 novel 하다`

이 세 가지는 현재 매트릭스만으로는 방어가 부족하다. 특히 `SLAug`, `StyCona`, `Stylizing ViT` 같은 일반 medical DG augmentation 계열 때문에, novelty는 `기법 자체`보다 `문제 재정의 + evaluation protocol + WBC-specific synthetic utility framing`에서 잡는 편이 안전하다.
