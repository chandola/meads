---
title: "Ethical AI @ UB Speaker Series"
layout: gridlay
excerpt: "Ethical AI @ UB Speaker Series"
sitemap: false
permalink: /speakers.html
---

# Speakers 
{% assign number_printed = 0 %}
{% for speaker in site.data.speakers %}

<div class="row">

<div class="col-sm-12 clearfix">
  <a href="{{ speaker.url }}"><img src="{{ site.url }}{{ site.baseurl }}/images/speakerpic/{{ speaker.photo }}" class="img-responsive" width="20%" style="float: left" /></a>
  <h4><a href="{{ speaker.url }}">{{ speaker.name }}</a></h4>
  <ul style="overflow: hidden">
  <i>{{ speaker.title }}</i>
  </ul>
  <ul style="overflow: hidden">
  <b>Title: </b>{{ speaker.talktitle }}
  <br/>
  <b>Abstract: </b>{{ speaker.talkabstract }}
  <br/>
  <b>Time: </b>{{ speaker.talktime }}
  <br/>
  <b>Location: </b>{{ speaker.talklocation }}
  </ul>
</div>

</div>
{% endfor %}
