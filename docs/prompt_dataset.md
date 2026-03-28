# 数据集 Prompt 列表 / Dataset Prompt Collection

本文档收录了用于训练 FLUX.2 Consistency LoRA 的约 60 条标注 prompt，覆盖多种艺术风格与场景，适用于艺术-to-写实图像配对数据集的构建。

---

## Prompt 格式说明

每条训练数据由一对图像和对应 caption 组成：

- **艺术图 caption**：描述艺术风格图像的内容，包含风格关键词
- **写实图 caption**：以 `MYCHAR` 为 trigger token，描述写实照片内容

---

## 一、动漫风格（Anime Style）共 15 条

| ID | 艺术图 Caption | 写实图 Caption |
|----|----------------|----------------|
| A01 | anime style portrait of a young woman, large eyes, light brown hair, gentle smile, cherry blossom background | MYCHAR photorealistic portrait, young woman, warm smile, cherry blossom bokeh background, soft natural lighting, 8k |
| A02 | anime girl standing in the rain, dark blue hair, school uniform, umbrella, night street | MYCHAR realistic photo, young woman standing in rain, dark hair, white shirt, holding umbrella, wet street reflections, cinematic |
| A03 | anime illustration, woman running through forest, green hair, long legs, sunlight filtering through trees | MYCHAR photorealistic, woman jogging in forest, natural light, dappled sunlight, dynamic pose, 4k |
| A04 | anime portrait, serious expression, silver hair, red eyes, warrior outfit, stormy sky | MYCHAR realistic warrior woman, silver hair, strong expression, dramatic sky, armor details, 8k photo |
| A05 | anime cute girl, pigtails, pink dress, sitting on bench, autumn leaves | MYCHAR photorealistic girl, pigtails, pink sundress, sitting on park bench, autumn foliage, warm tones |
| A06 | anime style, close-up face, ocean eyes, flowing black hair, sea breeze | MYCHAR cinematic portrait, woman with dark flowing hair, ocean in background, wind in hair, blue hour lighting |
| A07 | anime girl, summer beach scene, white bikini, tanned skin, surfboard | MYCHAR photorealistic, woman on beach, white swimsuit, natural tan, golden hour, ocean waves |
| A08 | anime magical girl, stars in hair, purple robe, moonlit night, floating | MYCHAR creative portrait, woman in purple dress, star accessories, moonlight, ethereal atmosphere |
| A09 | anime schoolgirl, textbook in hand, library interior, warm lamp light | MYCHAR realistic photo, young woman in library, holding book, cozy lighting, focused expression |
| A10 | anime art, woman in traditional kimono, red maple leaves, koi pond | MYCHAR photorealistic, woman wearing kimono, japanese garden, koi pond, autumn colors, ultra-detailed |
| A11 | anime portrait, woman with cat ears, casual outfit, coffee shop background | MYCHAR photo, young woman, cafe background, casual style, warm interior lighting, natural |
| A12 | anime girl, snowy landscape, fur coat, breath visible, winter forest | MYCHAR winter portrait, woman in fur coat, snowy forest, breath mist, cold light, 8k |
| A13 | anime art, female warrior, desert background, sand storm, battle pose | MYCHAR action portrait, woman in desert, dramatic wind, sand particles, dynamic composition |
| A14 | anime close-up, teary eyes, rain on window, melancholic mood | MYCHAR emotional portrait, woman by rain-streaked window, moody lighting, close-up, bokeh |
| A15 | anime style, festival scene, yukata dress, fireworks, summer night | MYCHAR photorealistic, woman in yukata, summer festival, fireworks background, warm night light |

---

## 二、油画风格（Oil Painting Style）共 15 条

| ID | 艺术图 Caption | 写实图 Caption |
|----|----------------|----------------|
| O01 | oil painting portrait, renaissance style, woman in red gown, candle light | MYCHAR classical portrait, woman in red dress, candlelit room, baroque style lighting, high detail |
| O02 | impressionist oil painting, woman in garden, flowers, dappled light | MYCHAR outdoor portrait, woman in floral garden, natural soft light, impressionist color palette |
| O03 | oil painting, female figure standing on cliff edge, dramatic ocean, storm | MYCHAR cinematic portrait, woman on coastal cliff, stormy sea, dramatic overcast sky |
| O04 | old master oil painting, seated woman, pearl necklace, interior | MYCHAR elegant portrait, woman seated, pearl jewelry, soft interior lighting, refined composition |
| O05 | romantic era oil painting, woman reading letter, window light, period dress | MYCHAR period portrait, woman reading, window light streaming in, vintage interior |
| O06 | oil painting, female musician, violin, concert hall, dramatic lighting | MYCHAR photo, woman playing violin, concert stage, dramatic spotlight |
| O07 | expressionist painting, distorted portrait, intense colors, raw emotion | MYCHAR expressive portrait, woman with intense gaze, bold lighting, striking composition |
| O08 | baroque oil painting, woman with fruit, opulent background, deep shadows | MYCHAR still life portrait, woman with fruit arrangement, moody chiaroscuro lighting |
| O09 | french impressionist painting, woman at cafe, street scene, afternoon light | MYCHAR candid street photo, woman at outdoor cafe, afternoon sun, Parisian atmosphere |
| O10 | oil portrait, female scientist, lab coat, bottles and equipment | MYCHAR professional photo, female scientist, laboratory setting, clean clinical lighting |
| O11 | oil painting landscape with figure, woman walking on country path, misty hills | MYCHAR landscape portrait, woman walking on rural path, morning mist, scenic vista |
| O12 | dutch golden age painting, woman spinning thread, domestic interior, window | MYCHAR documentary portrait, woman working, natural window light, simple interior |
| O13 | surrealist oil painting, woman floating, dreamlike landscape, impossible colors | MYCHAR surreal creative portrait, woman in floaty dress, conceptual background |
| O14 | oil painting, female dancer, stage curtains, spotlight, mid-pose | MYCHAR dance photography, female dancer on stage, dramatic spotlight, motion blur |
| O15 | classical oil painting, woman playing chess, intellectual expression, study | MYCHAR portrait, woman playing chess, thoughtful expression, cozy study room |

---

## 三、素描/铅笔风格（Sketch / Pencil Style）共 10 条

| ID | 艺术图 Caption | 写实图 Caption |
|----|----------------|----------------|
| S01 | pencil sketch, female portrait, crosshatching, paper texture, detailed eyes | MYCHAR photorealistic portrait, detailed eyes, soft studio lighting, clean background |
| S02 | charcoal drawing, woman's profile, strong contrast, dramatic shadows | MYCHAR side profile portrait, dramatic side lighting, high contrast, black and white |
| S03 | ink sketch, full body female figure, fashion pose, dynamic lines | MYCHAR fashion photography, full body pose, clean studio, fashionable outfit |
| S04 | technical pencil drawing, female face anatomy study, precise lines | MYCHAR close-up portrait, symmetrical face, clean neutral background, precision lighting |
| S05 | rough sketch, woman laughing, expressive lines, candid | MYCHAR candid portrait, woman laughing, genuine expression, natural warm light |
| S06 | graphite drawing, elderly woman, wrinkles, kind eyes, detailed texture | MYCHAR portrait, elderly woman, character face, soft window light, dignified |
| S07 | sketch study, female figure in motion, ballet pose, flowing lines | MYCHAR dance photography, ballerina mid-pose, stage lighting, graceful movement |
| S08 | pencil portrait, young woman, curly hair, freckles, natural look | MYCHAR natural portrait, young woman, curly hair, freckles, outdoor soft light |
| S09 | cross-hatched illustration, woman in coat, urban background, confident | MYCHAR street style photo, woman in coat, city background, confident pose |
| S10 | loose sketch, mother and child, tender moment, soft lines | MYCHAR lifestyle photo, mother and child, warm tender moment, natural light |

---

## 四、水彩风格（Watercolor Style）共 10 条

| ID | 艺术图 Caption | 写实图 Caption |
|----|----------------|----------------|
| W01 | watercolor painting, female portrait, soft edges, pastel colors, floral | MYCHAR soft portrait, woman, floral background, pastel tones, diffused light |
| W02 | watercolor illustration, woman with umbrella, rainy city street, bleeding colors | MYCHAR rainy street photo, woman with umbrella, reflective puddles, moody atmosphere |
| W03 | loose watercolor, beach scene, woman in hat, sun and sea, vibrant | MYCHAR beach portrait, woman in sun hat, ocean background, bright summer day |
| W04 | watercolor figure study, woman in flowing dress, wind, garden | MYCHAR outdoor portrait, woman in flowing dress, garden, breeze, candid |
| W05 | watercolor wedding portrait, bridal veil, soft romantic light | MYCHAR wedding photography, bride, soft romantic lighting, veil, floral bouquet |
| W06 | watercolor illustration, woman in autumn park, falling leaves, warm hues | MYCHAR autumn portrait, woman in park, falling leaves, warm golden light |
| W07 | watercolor food illustration, female chef, kitchen, colorful ingredients | MYCHAR culinary photo, female chef, bright kitchen, fresh ingredients, professional |
| W08 | watercolor travel sketch, woman exploring ancient ruins, backpack | MYCHAR travel photography, woman at ancient ruins, backpack, exploration mood |
| W09 | delicate watercolor, woman reading, cozy window seat, rainy day | MYCHAR lifestyle photo, woman reading by window, cozy interior, rain outside |
| W10 | watercolor portrait, girl with flowers in hair, spring garden, dew drops | MYCHAR spring portrait, girl with flower crown, garden setting, fresh morning light |

---

## 五、多场景叙事系列 Prompt（Narrative Series Prompts）共 10 条

以下 prompt 专为叙事系列生成设计，每组 3 张连续场景，展现角色在不同环境中的一致性表现。

### 系列一：雨后森林 → 沙漠星空 → 热带海滩

| 场景 | Prompt |
|------|--------|
| 场景 1 | MYCHAR photorealistic, woman walking through misty forest after rain, barefoot, flowing white dress, fog and green light, cinematic |
| 场景 2 | MYCHAR photorealistic, same woman lying on car roof in desert, starry night sky, milky way, warm lantern light, peaceful expression |
| 场景 3 | MYCHAR photorealistic, same woman relaxing on tropical beach lounger, sunset golden hour, gentle waves, white bikini, serene |

### 系列二：都市摩天楼 → 雪山顶峰 → 海底珊瑚

| 场景 | Prompt |
|------|--------|
| 场景 1 | MYCHAR photorealistic, woman on rooftop of skyscraper, urban night panorama, neon reflections, business casual, confident |
| 场景 2 | MYCHAR photorealistic, same woman on mountain summit, snow and ice, dawn light, expedition gear, triumphant |
| 场景 3 | MYCHAR photorealistic, same woman underwater, colorful coral reef, scuba gear, dappled light from surface, wonder expression |

---

## 六、负向 Prompt 参考

以下负向 prompt 适用于写实转换任务，帮助抑制艺术风格残留：

```
anime, cartoon, illustration, painting, sketch, drawing, watercolor, comic, 
manga, CGI, rendered, low quality, blurry, noisy, distorted, low resolution,
deformed, extra limbs, bad anatomy, watermark, signature, text, logo
```

---

## 七、推理阶段推荐 Prompt 模板

```python
# 正向 prompt 模板
POSITIVE_TEMPLATE = """
MYCHAR photorealistic portrait, {scene_description}, 
{lighting_description}, 8k ultra-detailed, sharp focus, 
professional photography, consistent identity
"""

# 负向 prompt 模板
NEGATIVE_TEMPLATE = """
anime, cartoon, painting, illustration, low quality, blurry, 
deformed anatomy, identity drift, inconsistent features
"""

# 示例使用
scene = "walking on tropical beach, sunset golden hour, gentle ocean breeze"
lighting = "warm golden hour backlight, soft shadows"
positive_prompt = POSITIVE_TEMPLATE.format(
    scene_description=scene,
    lighting_description=lighting
)
```
