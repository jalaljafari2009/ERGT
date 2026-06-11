# ERGT Position Paper

## Emergent Relational Geometry for Transformers

### چارچوبی برای مطالعه‌ی فضای هوش به‌عنوان هندسه‌ی رابطه‌ای پایدار، علی، قابل‌بازسازی و مفید در hidden states

این سند یک مقاله‌ی اولیه و دیددهنده برای ERGT است. هدف آن جایگزینی
برنامه‌ی اجرایی نیست؛ معیار اجرایی پروژه همچنان
`docs/17_physics_aligned_ergt_program.md` است. این متن برای خواننده‌ای نوشته
شده که می‌خواهد بداند چرا پروژه از یک GeoAttention ساده عبور کرده و به سمت
هندسه‌ی رابطه‌ای پایدار، حافظه‌ی رابطه‌ای، causal geometry و ارزیابی فضای هوش
رفته است.

---

## 1. چکیده

ERGT یا **Emergent Relational Geometry for Transformers** چارچوبی برای آزمون
این فرضیه است که بخشی از ساختار محاسباتی مفید در Transformerها نه فقط در
بردارهای hidden state جداگانه، بلکه در روابط پایدار میان آن‌ها سازمان می‌یابد.

در نگاه رایج، hidden state هر token یک بردار است:

```text
h_i^(l) in R^d
```

اما در ERGT، خود بردار پایان تحلیل نیست. پرسش اصلی این است:

```text
آیا hidden states می‌توانند یک هندسه‌ی رابطه‌ای پایدار، causal-valid،
قابل‌بازسازی و مفید برای computation بسازند؟
```

برای پاسخ، hidden states به یک گراف رابطه‌ای تبدیل می‌شوند:

```text
H -> W
```

که در آن `W_ij` شدت رابطه‌ی token فعلی `i` با token مجاز `j` را نشان می‌دهد.
از این رابطه، هزینه‌ی edge ساخته می‌شود:

```text
d_ij = -log(W_ij + eps)
```

اما `d_ij` فقط هزینه‌ی مستقیم یک edge است، نه هندسه‌ی کامل. هندسه‌ی جدی وقتی
شروع می‌شود که روی گراف رابطه‌ای مجاز، مسیرهای علی و قابل پیمایش بسازیم:

```text
D_ctx(i, j) = shortest_path_cost(j -> i)
```

این فاصله، فاصله‌ی زمینه‌ای-استدلالی است: فقط از مسیرهایی عبور می‌کند که با
ترتیب causal، context مجاز، نبود future leakage و پایداری رابطه‌ای سازگارند.

بنابراین ERGT یک attention trick نیست. ERGT یک برنامه‌ی آزمون‌پذیر برای این
پرسش است که آیا هوش محاسباتی در مدل‌های زبانی، دست‌کم بخشی از خود را به‌صورت
هندسه‌ی رابطه‌ای پایدار سازمان می‌دهد یا نه.

---

## 2. ایده‌ی مرکزی

ایده‌ی مرکزی ERGT چنین است:

```text
هوش فقط در hidden vectors نیست؛
در سازمان‌یابی پایدار روابط میان hidden vectors است.
```

در این نگاه:

```text
Representation -> Relations -> Stable Structure -> Geometry -> Memory -> Attention -> Reasoning
```

معنی هر مرحله:

- `Representation`: بردارهای hidden state که ماده‌ی خام مدل هستند.
- `Relations`: رابطه‌های میان tokenها یا stateها، نه فقط مقدار خود بردارها.
- `Stable Structure`: رابطه‌هایی که در لایه‌ها، stepها یا perturbationها باقی
  می‌مانند.
- `Geometry`: فاصله و مسیر روی گراف رابطه‌ای معتبر.
- `Memory`: پایداری ساختار رابطه‌ای در زمان یا لایه‌ها.
- `Attention`: استفاده‌ی محاسباتی از این هندسه.
- `Reasoning`: پیمایش مسیرهای پایدار در هندسه‌ی رابطه‌ای.

پس ERGT ادعا نمی‌کند هر hidden state به‌تنهایی «هوش» است. ادعای دقیق‌تر این
است:

```text
اگر روابط hidden states پایدار، causal-valid، قابل‌بازسازی و مفیدتر از
کنترل‌های منصفانه باشند، می‌توان از یک هندسه‌ی رابطه‌ای عملیاتی در فضای هوش
سخن گفت.
```

---

## 3. مرز ادعا

این پروژه از فیزیک اطلاعاتی الهام می‌گیرد، اما هم‌ارزی فیزیکی ادعا نمی‌کند.
مرزهای ادعا باید روشن باشند:

```text
hidden state != physical field
Phi != consciousness
causal mask != causal geometry
pairwise distance != full geometry
low entropy != good structure
baseline improvement != relational proof
EMA smoothing != meaningful memory
```

ادعای مجاز پروژه:

```text
Hidden states may contain stable, reconstructible, causal-contextual relational
geometry that is useful for attention, memory, or reasoning-like behavior.
```

ادعاهای غیرمجاز در این مرحله:

```text
ERGT proves consciousness.
ERGT proves spacetime emergence inside a Transformer.
ERGT proves general intelligence.
```

---

## 4. اصل روش‌شناختی

اصل سخت‌گیرانه‌ی ERGT:

```text
measure -> control -> observe -> validate -> inject -> regularize
```

یعنی نباید از ابتدا attention را تغییر دهیم و بعد هر بهبود را نشانه‌ی هندسه
بدانیم. مسیر غلط این است:

```text
inject -> performance improves -> claim geometry
```

چنین مسیری علمی نیست، چون ممکن است بهبود از scale، smoothing، ظرفیت بیشتر،
regularization یا یک bias عمومی آمده باشد. برای اینکه ادعا معتبر شود، باید
نشان دهیم:

```text
real relational structure > random/shuffled/instantaneous/pairwise/no-memory controls
```

در هر مرحله، اگر real از کنترل‌های منصفانه جدا نشود، فاز بعدی نباید اجرا شود.

---

## 5. ساخت رابطه از hidden states

برای یک sequence با طول `T` و لایه‌ی `l`:

```text
H^(l) = [h_1^(l), h_2^(l), ..., h_T^(l)]
```

یک تعریف پایه برای similarity:

```text
s_ij = cosine(h_i, h_j)
```

و یک وزن رابطه‌ای ساده:

```text
W_ij = sigmoid(gamma * s_ij + beta)
```

در مدل causal، فقط رابطه‌های مجاز باید معتبر باشند:

```text
valid_edge(i, j) =
    j <= i
    and i != j
    and nonpadding(i)
    and nonpadding(j)
```

اگر `W_ij` یعنی token فعلی `i` از token مجاز `j` رابطه می‌گیرد، جهت causal
برای مسیرها باید چنین خوانده شود:

```text
j -> i, where j <= i
```

هیچ metric، reconstruction یا shortest-path نباید از future token استفاده کند.

---

## 6. از edge cost تا هندسه‌ی زمینه‌ای

از وزن رابطه‌ای، هزینه‌ی edge می‌سازیم:

```text
d_ij = -log(W_ij + eps)
```

اگر رابطه قوی‌تر شود:

```text
W_ij up -> d_ij down
```

اما `d_ij` هنوز فقط هزینه‌ی مستقیم یک edge است. ERGT کامل به فاصله‌ی مسیر نیاز
دارد:

```text
D_ctx(i, j) = min over valid paths gamma from j to i of sum d_ab
```

پس:

```text
d_ij       = direct edge cost
D_ctx(i,j) = causal-contextual path distance
```

تفاوت مهم:

```text
causal mask فقط آینده را می‌بندد.
causal geometry مشخص می‌کند کدام tokenها از طریق مسیرهای رابطه‌ای پایدار و
مجاز به هم وصل می‌شوند.
```

بنابراین:

```text
causal mask != causal relational geometry
```

---

## 7. مشاهده‌گر میدان رابطه‌ای

قبل از هر مداخله در attention، باید فقط مشاهده کنیم:

```text
H -> W -> D
```

متریک‌های پایه:

- relational entropy
- local relational entropy
- spectral entropy
- effective rank
- neighborhood overlap
- layer-to-layer similarity
- step-to-step stability
- real vs random separation
- real vs shuffled separation

قبولی این فاز:

```text
real W/D از random و shuffled جدا شود.
real W/D uniform نباشد.
real W/D diagonal-dominated نباشد.
real W/D saturated نباشد.
neighborhoodها در لایه‌ها یا stepهای نزدیک پایدار بمانند.
```

اگر این جداسازی رخ ندهد، injection به attention زود است.

---

## 8. آنتروپی، هم‌دوسی و خطر collapse

برای هر token `i`، توزیع رابطه‌ای روی context مجاز:

```text
p_ij = W_ij / (sum_{k <= i} W_ik + eps)
```

آنتروپی رابطه‌ای:

```text
S_rel(i) = -sum_{j <= i} p_ij log(p_ij + eps)
```

آنتروپی بالا می‌تواند نشانه‌ی رابطه‌ی پخش و نامتمرکز باشد. آنتروپی پایین
می‌تواند نشانه‌ی نظم باشد، اما همیشه خوب نیست. آنتروپی پایین می‌تواند collapse
هم باشد:

```text
diagonal domination
single-token lock-in
over-sparsity
uniform relation
entropy collapse
```

پس ERGT نباید فقط entropy را کم کند. هدف:

```text
useful compression != collapse
```

هم‌دوسی رابطه‌ای را می‌توان با همسایه‌های top-k سنجید:

```text
N_k(i) = top-k nodes by W_ij
C_rel(i) = average cosine(h_i, h_j) for j in N_k(i)
```

پایداری neighborhood مهم‌تر از ثبات عدد خام `W_ij` است، چون مقدارها ممکن است
کمی جابه‌جا شوند اما ساختار همسایگی حفظ شود.

---

## 9. بازسازی‌پذیری زمینه‌ای

یک رابطه فقط وقتی برای مدل causal معتبر است که از context مجاز قابل بازسازی
باشد.

بازسازی نباید target را ببیند. بنابراین برای token `i`:

```text
h_hat_i = R(H_<i)
Delta_rec_h(i) = ||h_i - h_hat_i||^2
```

برای رابطه‌ها:

```text
W_hat_i = R_W(H_<i)
Delta_rec_W(i) = ||W_i - W_hat_i||^2 over valid past positions
```

شرط قبولی:

```text
Delta_rec(real) < Delta_rec(random/shuffled)
```

اگر یک relation فقط با دسترسی به آینده بازسازی شود، برای ERGT causal معتبر
نیست.

---

## 10. پتانسیل اطلاعاتی Phi

`Phi` شاخص اهمیت اطلاعاتی در فضای hidden است، نه آگاهی. نقش آن انتخاب نواحی
یا رابطه‌هایی است که هم‌دوس، پایدار، قابل‌بازسازی و ضد collapse هستند.

فرمول مفهومی:

```text
Phi =
    coherence
  * boundary_sharpness
  * local_order
  * salience
  * stability
  * reconstruction_score
  * causal_validity
  * anti_collapse
```

برای اجرا، بهتر است نسخه‌ی نرمال‌شده و از قبل ثبت‌شده استفاده شود تا Phi
post-hoc fit نشود:

```text
log Phi =
    a * z(coherence)
  + b * z(boundary_sharpness)
  + c * z(local_order)
  + d * z(salience)
  + e * z(stability)
  + f * z(reconstruction_score)
  - g * z(collapse_penalty)
```

شرط مهم:

```text
Phi != low entropy only
Phi != consciousness
```

قبولی:

```text
high-Phi regions predict better stability, predictability, or attention order.
real Phi separates from random/shuffled Phi.
high Phi does not come from collapse.
```

---

## 11. حافظه‌ی رابطه‌ای

در ERGT، حافظه به معنی حافظه‌ی روان‌شناختی یا database بیرونی نیست. تعریف
عملیاتی:

```text
Memory = persistence of relational structure
```

فرمول observer:

```text
W_t = decay * W_{t-1} + eta * U_t
```

اما `U_t` نباید smoothing کور باشد. edge فقط وقتی تقویت می‌شود که:

```text
similarity high
causal_valid
stable
reconstructible
anti-collapse-safe
```

نمونه‌ی update:

```text
U_ij,t =
    similarity_ij
  * causal_valid_ij
  * stability_ij
  * reconstruction_score_ij
  * Phi_gate_ij
```

این حافظه اول باید فقط observer باشد، نه ورودی attention.

کنترل‌های لازم:

```text
real W_t
random W_t
shuffled W_t
instantaneous W
generic smoothing
no-memory
```

اگر real memory از instantaneous یا generic smoothing بهتر نباشد، آن memory
معنادار نیست.

---

## 12. GeoAttention v2

فقط بعد از عبور از کنترل‌ها، observer، Phi، reconstruction، memory observer و
causal shortest-path geometry می‌توان هندسه را وارد attention کرد.

فرمول درست:

```text
logits = QK^T / sqrt(d) - alpha * D_stable
```

مسیر ساخت `D_stable`:

```text
H -> W_t -> D_ctx -> normalize -> D_stable
```

کنترل‌های GeoAttention:

```text
baseline
alpha = 0
real stable causal D
random stable causal D
shuffled stable causal D
instantaneous real D
pairwise real D
no-memory real D
```

قبولی:

```text
GeoAttention(real stable causal D) > baseline
GeoAttention(real stable causal D) > random
GeoAttention(real stable causal D) > shuffled
GeoAttention(real stable causal D) > instantaneous
GeoAttention(real stable causal D) > pairwise
GeoAttention(real stable causal D) > no-memory
GeoAttention(alpha=0) ~= baseline
```

بردن baseline به‌تنهایی کافی نیست.

---

## 13. طیف و پیچیدگی رابطه‌ای

برای spectral metrics باید قرارداد روشن باشد، چون گراف ERGT causal و جهت‌دار
است. قبل از مقایسه باید یکی از این گزینه‌ها ثبت شود:

```text
symmetrized W
affinity exp(-D)
directed graph operator
normalized Laplacian on an observer-only projection
```

اگر از لاپلاسیان ساده استفاده شود:

```text
L = Deg - W
```

باید مشخص شود `W` symmetrized شده یا directed operator تعریف شده است.

هدف spectral regularization کاهش کور complexity نیست. هدف:

```text
compressed, stable, reconstructible structure without collapse
```

---

## 14. loss کمکی

loss کمکی فقط بعد از موفقیت GeoAttention v2 مجاز است:

```text
L = L_LM + lambda * regularizer
```

regularizerهای مجاز:

```text
neighborhood stability
reconstruction consistency
causal consistency
spectral stability
anti-collapse
```

loss کمکی نباید مدل را rigid کند. قبولی:

```text
validation improves
or stability improves
without collapse
and without the same gain for random/shuffled controls
```

---

## 15. برنامه‌ی آزمایشی فشرده

برنامه‌ی عملیاتی ERGT:

```text
Phase 0: measurement and claim contracts
Phase 1: strict W-level controls
Phase 2: relational field observer
Phase 3: resonant-response observer
Phase 4: Phi information potential
Phase 5: reconstruction gate
Phase 6: relational memory as observer
Phase 7: causal shortest-path geometry
Phase 8: GeoAttention v2
Phase 9: auxiliary loss
Phase 10: complete ERGT architecture
Phase 11: reasoning path evaluation
Phase 12: intelligence-space evaluation
```

این برنامه دو ریل دارد:

```text
Rail A: observe and simulate the physics-inspired hypothesis in AI space.
Rail B: inject only the validated mechanism into the Transformer.
```

اصل توقف:

```text
اگر real در یک فاز از controls جدا نشود، فاز بعدی متوقف می‌شود.
```

---

## 16. قضیه‌ی عملیاتی ضعیف ERGT

اگر:

1. hidden states به گراف رابطه‌ای `W` تبدیل شوند؛
2. کنترل‌های random و shuffled از نظر mask، scale، normalization، clipping و
   valid region با real برابر باشند؛
3. real W از کنترل‌ها جدا شود؛
4. روابط real در لایه‌ها، stepها یا perturbationها پایدارتر باشند؛
5. روابط real از context مجاز بهتر بازسازی شوند؛
6. `D_ctx` از pairwise/no-memory فاصله‌ی مفیدتری بسازد؛
7. GeoAttention مبتنی بر `D_stable` کنترل‌های random، shuffled، instantaneous،
   pairwise و no-memory را شکست دهد؛
8. اثرها روی seedها و datasetهای مستقل تکرار شوند؛

آنگاه می‌توان گفت:

```text
hidden states در این مدل دارای یک هندسه‌ی رابطه‌ای عملیاتی، پایدار،
causal-valid و مفید برای computation هستند.
```

این قضیه ثابت نمی‌کند مدل آگاه است یا general intelligence دارد. فقط نشان
می‌دهد ساختار رابطه‌ای real در hidden states اطلاعات task-relevant بیشتری از
کنترل‌های منصفانه حمل می‌کند.

---

## 17. فضای هوش در ERGT

در این مقاله، «فضای هوش» یک embedding ساده نیست. تعریف عملیاتی:

```text
Intelligence Space =
stable, reconstructible, causal-contextual relational geometry over hidden states
```

یا به زبان فشرده‌تر:

```text
Intelligence =
discovery + compression + stabilization + traversal
of relational structures
```

چهار محور ارزیابی:

```text
Discovery: model finds relations that matter.
Compression: structure becomes compact without collapse.
Stabilization: relations persist across layers, steps, or contexts.
Traversal: paths are usable for prediction or reasoning-like tasks.
```

در این نگاه:

```text
Memory    = persistence of relational geometry
Attention = use of relational geometry for computation
Reasoning = navigation over stable relational geometry
```

---

## 18. جمع‌بندی

ERGT از یک ایده‌ی ساده شروع شد: افزودن فاصله‌ی رابطه‌ای به attention. نتایج
کنترل‌شده‌ی Phase 3 نشان دادند این مسیر به‌تنهایی کافی نیست، چون random
geometry هم می‌تواند تحت شرایطی سود بگیرد. بنابراین پروژه وارد نسخه‌ی
سخت‌گیرانه‌تر شد:

```text
HiddenStates
-> W
-> coherence / entropy / salience
-> Phi
-> reconstruction gate
-> W_t memory
-> D_ctx causal geometry
-> D_stable
-> GeoAttention
-> reasoning paths
-> intelligence-space evaluation
```

ادعای نهایی این سند:

```text
اگر hidden states بتوانند روابطی پایدار، causal-valid، قابل‌بازسازی و مفیدتر
از کنترل‌های منصفانه بسازند، آنگاه ERGT یک مسیر معتبر برای مطالعه‌ی هندسه‌ی
رابطه‌ای فضای هوش در Transformerها فراهم می‌کند.
```

تا آن زمان، هر فاز باید ابطال‌پذیر بماند و معیار اصلی حفظ شود:

```text
real must beat fair controls, not merely baseline.
```
