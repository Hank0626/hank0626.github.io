---
layout: page
permalink: /publications/
title: Publications
description: 
nav: true
nav_order: 1
---

<p>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
    tex2jax: {inlineMath: [['\\(','\\)']]}
    });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.min.js" integrity="sha384-R4Ooq3D9I1no8pt7thk3GSDB2X5A/eWYwJ8HfpWiZjbbKzLjoFDjFYZE5bFBfvsD" crossorigin="anonymous"></script>

<em style="color: #808000;">\(^*\) Equal contribution. \(^\dagger\) Corresponding author.</em>
</p>

<!-- _pages/publications.md -->
<div class="publications">

{% bibliography -f {{ site.scholar.bibliography }} %}

</div>
