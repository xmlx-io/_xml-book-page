[![GitHub Release](https://img.shields.io/github/v/release/xmlx-io/xml-book?display_name=tag&logo=github)](https://github.com/xmlx-io/xml-book/releases/latest)  
[![Read Book][book-badge]](https://book.xmlx.io)
[![Read Docs](https://img.shields.io/badge/read-docs-blue.svg?logo=gitbook)](https://book.xmlx.io/docs)  
[![Licence](https://img.shields.io/badge/licence-CC%20BY--NC--SA%204.0-red)](LICENCE)
[![CLA](https://img.shields.io/badge/CLA-Apache%202.2-red)](CLA.md)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)  
[![DOI](https://zenodo.org/badge/DOI/XX.XXXX/zenodo.XXXXXXX.svg)](https://doi.org/XX.XXXX/zenodo.XXXXXXX)
[![Cite BibTeX](https://img.shields.io/badge/cite-bibtex-yellow.svg)](https://book.xmlx.io/docs/#citing-the-book)

# :books: eXplainable Machine Learning :books: #

The [`master`][master] branch of this repository holds the source of our
eXplainable Machine Learning book (Jupyter Book)
and its documentation (MkDocs) --
[`book`](book) and [`docs`](docs) folders respectively.
These sources are built into a collection of static HTML documents served by
GitHub Pages from the [`gh-pages`][gh_pages] branch.
The book can be accessed at [book.xmlx.io][book]
and the documentation at [book.xmlx.io/docs][docs].

General information about our organisation can be found at [xmlx.io][org].
Our resources are hosted on GitHub under the [xmlx-io][gh_org] organisation.

## :book: Book ##

[![Read Book][book-badge]](https://book.xmlx.io)

The interactive Jupyter Book version of our XML book can be accessed at
[book.xmlx.io][book].
It covers a broad range of eXplainable Machine Learning theory and practice,
including:

* high-level overviews & introductory examples;
* mathematical foundations;
* algorithmic implementations;
* practical advice & real-life caveats; and
* success & failure case studies.

## :memo: Documentation ##

[![Read Docs](https://img.shields.io/badge/read-docs-blue.svg?logo=gitbook)](https://book.xmlx.io/docs)

The MkDocs documentation of our XML book can be accessed at
[book.xmlx.io/docs][book].
It covers technical concepts to do with how the book is built
and maintained, as well as a *contributor's guide* that outlines how to
author new content for the book.
In particular, you will find there a *code of conduct*, *changelog*,
ways to *contact us* and *acknowledgements*.
The documentation also includes a dedicated page describing our preferred
*contribution workflow* and information how to set up your
*development environment*,
both for working with the book and the documentation itself.

[master]: https://github.com/xmlx-io/xml-book/tree/master
[gh_pages]: https://github.com/xmlx-io/xml-book/tree/gh-pages
[gh_org]: https://github.com/xmlx-io
[org]: https://xmlx.io/
[book]: https://book.xmlx.io/
[docs]: https://book.xmlx.io/docs
[book-badge]: https://img.shields.io/badge/read-book-orange.svg?logo=data:image/svg+xml;base64,PHN2ZyBpZD0iTGF5ZXJfMSIgZGF0YS1uYW1lPSJMYXllciAxIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA4MS43OCA3MS4xNSI+PGRlZnM+PHN0eWxlPi5jbHMtMXtmaWxsOiNmZTdkMzc7fS5jbHMtMntmaWxsOiNmZmY7fTwvc3R5bGU+PC9kZWZzPjx0aXRsZT5sb2dvLXNxdWFyZTwvdGl0bGU+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNNC44LDQ1Ljg3cTIuNyw4LjQ2LDEzLjc4LDguNzVoOS4xN3E5LjYzLDAsMTMuMTMsNy41M1E0NC4zLDU0LjYyLDU0LDU0LjYyaDguNDVxMTEuNzYsMCwxNC41NS04LjcybDQuNzgsMS42UTc5LjcxLDU0Ljg4LDc1LDU4LjU4VDYyLjYzLDYyLjMzSDUzLjg4cS0xMCwwLTEwLDguODJoLTZxMC04LjgyLTEwLjEzLTguODJIMTkuMzNxLTcuODMsMC0xMi41My0zLjc4VDAsNDcuNVoiLz48cGF0aCBjbGFzcz0iY2xzLTIiIGQ9Ik0yMC4zOSw0MS4zNEExMy44OCwxMy44OCwwLDAsMSwxNC4yMyw0MGExMC41OSwxMC41OSwwLDAsMS00LjQyLTQuMDZBMTQuMTcsMTQuMTcsMCwwLDEsOCwyOS4xNEw4LDI5aDUuM2MuMDcsMi43NC43LDQuNzUsMS44OCw2LjA1YTYuNTgsNi41OCwwLDAsMCw1LjA5LDEuOTIsNi43Myw2LjczLDAsMCwwLDMuOC0xLDUuOSw1LjksMCwwLDAsMi4yLTIuNzUsMTAuNDMsMTAuNDMsMCwwLDAsLjcyLTRWNC4zM2wtNS40NC0uNzJWMEgzNy4xMVYzLjU1bC00LjU3Ljc4VjI5LjE5YTE0LDE0LDAsMCwxLTEuMzksNi4zOSw5LjkyLDkuOTIsMCwwLDEtNC4wOSw0LjI1QTEzLjQzLDEzLjQzLDAsMCwxLDIwLjM5LDQxLjM0Wm0yMC45MS0uNTlWMzcuMjJsNC41Ni0uNzhWNC4zM0w0MS4zLDMuNTVWMEg1OC44OHE2LjM5LDAsMTAsMi43NWMyLjQsMS44MywzLjU5LDQuNTksMy41OSw4LjI3YTcuNTksNy41OSwwLDAsMS0xLjcxLDQuODYsMTAuMzcsMTAuMzcsMCwwLDEtNC41NSwzLjE4LDkuNTMsOS41MywwLDAsMSw0LjIyLDIsMTAsMTAsMCwwLDEsMi43MywzLjU4LDEwLjkyLDEwLjkyLDAsMCwxLDEsNC42NHEwLDUuNTktMy42NCw4LjU1dC05Ljg2LDNabTEwLjA4LTQuMzFoOS4yM2E4LjUsOC41LDAsMCwwLDUuODYtMS44Niw2LjY4LDYuNjgsMCwwLDAsMi4wOS01LjI4LDEwLDEwLDAsMCwwLS43Ni00LjExLDUuNjQsNS42NCwwLDAsMC0yLjM2LTIuNjMsNy42NSw3LjY1LDAsMCwwLTQtLjkzSDUxLjM4Wm0wLTE5LjExaDguOTNhNi43LDYuNywwLDAsMCw0Ljc5LTEuNzIsNi4xNyw2LjE3LDAsMCwwLDEuODQtNC43QTUuODksNS44OSwwLDAsMCw2NC44Niw2YTkuMzgsOS4zOCwwLDAsMC02LTEuNjRoLTcuNVoiLz48L3N2Zz4K
