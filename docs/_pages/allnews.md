---
title: "News"
layout: textlay
excerpt: "Ethical AI at UB."
sitemap: false
permalink: /allnews.html
---

# News

{% for article in site.data.news %}
<p>{{ article.date }} <br>
<em>{{ article.headline }}</em>
{{ article.description }}</p>
{% endfor %}
