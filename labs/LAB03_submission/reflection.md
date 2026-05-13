
| Genre | Part A Accuracy | Part B Accuracy |
|---|---|---|
| Animation | 0.880 | 0.807 |
| Comedy | 0.733 | 0.700 |
| Documentary | 0.827 | 0.827 |
| Horror | 0.673 | 0.627 |
| Romance | 0.587 | 0.587 |
| Sci-Fi | 0.560 | 0.627 |
| **Overall** | **0.710** | **0.696** |

Then address these questions:

1. **Architecture choices**: Describe the image branch and tabular branch architectures you settled on. Why did you choose this structure? What did you try that didn't work as well?
For the image branch, I settled on 4 convolutional blocks each with a Conv2d layer, BatchNorm, ReLu and MaxPool. The blocks progressivley expand channel depth while having spatial resolution at each step. Global average pooling then collapses the spatial dimensions to a 128 dimensional vector followed by dropout and a linear projection to 256 dimensions. The jey design choice was using BatchNorm after eveyr conv layer. Global average pooling was better than flattening because it reduces the parameter count and is more robust to small spatial shifts. 
For the tabular branch, I used 2 parallel sub-branches that merge before the output. The numeric sub-branch runs the 7 standardized features through 2 FC layers with dropout between them. The embedding sub-branch creates one nn.Embedding table per categorical field, mean-pools each field's token embeddings while masking out padding positions, concatenates all pooled vectors and projects through a single Fc layer to 128 dimensions. The two 128-dim vectors are contencated and projected to 256 dimesnsions as the final tabular output. 

2. **Overfitting**: Did you observe a gap between training and validation accuracy? At what point did it appear? What strategies did you use to combat it (dropout, weight decay, early stopping, smaller vocabulary, reduced model size, learning rate scheduling)? Which were most effective?
A gap between training and validation accuracy appeared by epoch 3 and widened gradually through the end of the training. By epoch 20, training accuracy reached 0.724 while the best validation accuracy was 0.731 — the gap was modest and validation accuracy continued improving through epoch 14, which suggests the regularization was pretty effective.

Three strategies were used together: dropout (0.4 in the image head and fusion head, 0.3 in the tabular branch), weight decay (1e-3 via Adam's `weight_decay` parameter), and cosine annealing for the learning rate schedule. Of these strategies, weight decay appeared to be the most useful. It consistently flattened the training loss curve without degrading validation performance, keeping the two curves close together throughout training. Dropout helped prevent the fusion head from memorizing training patterns in the later epochs. The cosine schedule reduced the likelihood of the optimizer overshooting a good minimum in late epochs, though the effect was hard to isolate given everything was trained jointly.

3. **Part A vs. Part B**: How did your custom CNN compare to the pretrained ResNet18? Did transfer learning help, and if so, in what way (higher accuracy, faster convergence, less overfitting)?
Part A outperformed Part B on the test set (71.0% vs. 69.6%), which I did not expect. The gap is small, but it was consistent across most genres. Documentary was the only genre where Part B matched Part A exactly.

The most likely explanation is that the frozen ResNet18 backbone, pretrained on ImageNet object classification, produces features optimized for identifying real-world objects and textures, not graphic design elements like title typography, color palettes, and stylized illustrations that dominate movie posters. The custom CNN trained directly on posters was free to learn those genre-relevant visual patterns from the start. With a fully frozen backbone, Part B's only trainable image-side parameters were the 128K in the projection head, which is a much smaller capacity than the full custom CNN.

Part B did converge at a comparable pace. Both models reached roughly 70% validation accuracy by epoch 10, so transfer learning didn't hurt convergence speed in a meaningful way. The result suggests that for highly domain-specific visual inputs like stylized poster art, training from scratch on that domain can match or beat a frozen ImageNet backbone, especially when the dataset is large enough (~7K training samples).

4. **Tabular branch insights**: Which metadata features seemed most useful for genre prediction? Look at the per-class accuracy table — which genres did the model struggle with most? Does that make sense given the available features? If you tried ablations (tabular-only or image-only), what did you learn?
The numeric features that likely contributed most were `vote_average`, `release_year`, and `runtime`. Documentaries tend to have above-average critical reception and shorter runtimes; animation skews toward certain release eras and has a tight runtime distribution. Budget and revenue were noisier signals because many films in the dataset have both recorded as zero (indicating missing data rather than true zeros), though the NumericScaler's mean imputation reduced the damage.

Looking at the per-class accuracy table, Romance and Sci-Fi were the hardest genres in Part A (0.587 and 0.560 respectively). This is consistent with what you'd expect from the available features. Romance posters share a lot of visual vocabulary with Comedy (soft color palettes, close-up portraits, similar typography styles) and the metadata doesn't clearly separate them either, since runtime and ratings distributions overlap substantially. Sci-Fi improved more in Part B (0.560→0.627), which hints that ResNet's edge and texture detectors may capture space imagery and dark dramatic lighting better than the custom CNN's shallower feature hierarchy.

Animation was the strongest genre in both parts (0.880 / 0.807). This makes sense, animation posters have a visually distinctive style (flat color fills, stylized characters, bright saturated palettes) and are often produced by a recognizable set of production companies, giving the tabular branch a clean signal as well.

5. **What would you do differently?** If you had more compute time or training data, what would you try next?
With more compute, the first thing to try would be fine-tuning the ResNet backbone's last residual block (layer4) after the head has converged, using a backbone learning rate roughly 10× lower than the head. The frozen-backbone Part B result suggests the ImageNet features don't perfectly align with poster art, and nudging the deepest layers to adapt could close the gap with Part A.

With more training data, increasing `TOP_N_VOCAB` from 50 to 200–500 would let the model learn meaningful embeddings for mid-frequency cast members and production companies rather than mapping most tokens to UNK. Right now, a film starring a moderately well-known actor (say, outside the top 50 by training frequency) gets the same UNK embedding as a completely unknown name, which wastes a real signal.

Two other things worth trying: a two-stage training schedule (train tabular-only for a few epochs to warm up the embeddings before introducing image gradients), and label smoothing in the cross-entropy loss to reduce overconfidence on the easiest genres like Animation, which might free capacity for the harder ones.

6. *(Optional — only if you completed optional extensions)* **Optional extensions**: For each optional experiment you ran, briefly describe what you tried, what result you got, and how it compared to your Part A baseline.