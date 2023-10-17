# Weak Supervision with Snorkel: Image Classification Example

[See my blog post for full description](https://ramintoosi.ir/posts/2023/08/blog-post-1/).

## Summary

In the world of machine learning, data is often hailed as the crown jewel that powers models and drives innovation. Yet, obtaining high-quality, labeled data remains a significant challenge, often demanding painstaking manual efforts from human annotators. This is where the concept of weak supervision emerges as a beacon of hope for machine learning engineers and practitioners.

Weak supervision is the art of leveraging various sources of noisy or imprecise supervision to label a large amount of data efficiently. It takes the burden off exhaustive manual labeling and opens the door to scaling up projects that might have been otherwise resource-intensive. In this post, we embark on a journey to explore the Snorkel, a powerful tool that empowers us to automate the labeling process, saving time and effort without compromising on results.

In this tutorial, tailored for machine learning engineers and enthusiasts alike, we’ll unveil the advantages of weak supervision using a practical example: Image Classification. By the end of this guide, you’ll have a basic understanding of how to harness the potential of Snorkel to streamline your image classification pipelines and achieve impressive results with reduced labeling efforts.

## Quick run

Run the code in the following steps

- Download necessary files.
```commandline
sh oiv7_files_download.sh
python oiv7_downloader.py
```
- Split downloaded images.
```commandline
python prepare_dataset.py
```
- Prepare noisy labels with Snorkel.
```commandline
python prepare_data_snorkel.py
```
- Train and test.
```commandline
python main.py
```

## References
- [Snorkel](https://github.com/snorkel-team/snorkel)
- [Open Images Downloader](https://raw.githubusercontent.com/openimages/dataset/master/downloader.py)