---
title: "Manifolds for Extreme-scale Applied Data Science (MEADS) - News"
layout: homelay
excerpt: "MEADS"
sitemap: false
permalink: /allnews.html
---

# News

{% for article in site.data.news %}
<p>{{ article.date }} <br>
<em>{{ article.headline }}</em>
{{ article.description }}</p>
{% endfor %}
