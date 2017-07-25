# ARAE
tensorflow implementation of [Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223) (ARAE)

While the Paper used the Stanford Natural Language Inference dataset for the text generation, this implementation only used the mnist dataset.
I implemented the continuous version for this implementaion, but discrete version is implemented as footnote in this code.

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/map.png width="1024"/>
</p>

## Dependencies

1. tensorflow == 1.0.0
1. numpy == 1.12.0
1. matplotlib == 1.3.1

## Steps

Run the following code for image reconstruction.

<pre><code>
python train.py
</code></pre>

## Results

- The model trained 100000 steps

- Generated from fake(noise) data
<p>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/fake_76500.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/fake_91500.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/fake_99500.png" width="112"/>
</p>
The result from noise tend to appear several figures simultaneously.

- Generated from real data
<p>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
  <img src="https://raw.githubusercontent.com/soobin3230/ARAE/master/png/real_42000.png" width="112"/>
</p>

## Notes

I didn't multiply the critic gradient before backpropping to the encoder.
