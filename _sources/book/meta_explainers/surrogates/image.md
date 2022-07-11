(text:meta-explainers:surrogates:image)=
# Image Surrogates #

(text:meta-explainers:surrogates:image:interpretable-representation)=
## Interpretable Representation ##

The interpretable representation of image data is built upon the same premise: images are algorithmically segmented into super-pixels, often using edge-based methods~\cite{ribeiro2016why} such as quick shift~\cite{vedaldi2008quick}. % operates similarly -- see Figure~\ref{fig:img_ex}. % is based on % stems from % whose ... and
Next, the presence (\(1\)) or absence (\(0\)) of these segments is manipulated by the underlying binary representation, where an all-\(1\) vector corresponds to the original picture -- see Figure~\ref{fig:img} for a reference. % representation % vector % with and % demonstration
% Next, the segments are represented as a binary vector indicating presence (\(1\)) or absence (\(0\)) of information in each super-pixel, where an all-\(1\) vector corresponds to the original picture. % with XXX assigned % image
% However, removing a super-pixel from an image when setting one of the interpretable components to \(0\) is an ill-defined procedure. %
However, since a segment of an image cannot be directly removed -- in contrast to the equivalent operation in text IRs -- setting one of the interpretable components to \(0\) is an ill-defined procedure. %
% Impossibility of Information Removal
Instead, a computationally-feasible proxy is commonly used to hide or discard the information carried by the super-pixels, namely, segments are occluded with a solid colour. %, which
For example, LIME uses the mean colour of each super-pixel to mask its content~\cite{ribeiro2016why}. % segment
Explanations based on such interpretable representations communicate the influence of each image segment on the black-box prediction of a user-specified class as shown in Figure~\ref{fig:img_ex}.% setting % image IRs % In this scenario % Regardless, % particular

This approach, nonetheless, comes with its own implicit assumptions and limitations, which are often overlooked. % masking % issues % contained in % However, % as we will explore later
For one, an edge-based partition of an image may not convey concepts that are meaningful from a human perspective. % correspond to % but % (cognitively) % However, the edge-based partition resulting partition
\emph{Semantic segmentation} or outsourcing this task to the user appears to yield better results~\cite{sokol2020limetree,sokol2020one}, possibly at the expense of automation difficulties. % usually % human-in-the
Additionally, the information removal proxy could be improved by replacing colour-based occlusion of super-pixels with a more meaningful process that better reflect how humans perceive visual differences between images. % two scenes % instead of replacing segment colouring with % segments
For example, the content of a segment could be occluded with another object, akin to Benchmarking Attribution Methods~\cite{yang2019bam}, or retouched in a context-aware manner, e.g., with what is anticipated in the background, thus preserving the integrity and colour continuity of the explained image. % natural % truly % ``magic brush'' % to be % expected % (BAM) % The most appealing and semantically-meaningful solution would be to ``remove'' by occluding it with another object ... retouching it
While both of these approaches are intuitive, they are difficult to automate and scale since the underlying operations are mostly limited to image partitions where each super-pixel represents a self-contained and semantically-coherent object.%, or their parts when blending them with adjacent segments is conceptually meaningful.% methods % of these % whole % but % they are
% , with the most viable solution being

\begin{figure}[t]
  \centering
  \begin{subfigure}[t]{.99\textwidth}
      \centering
      % diagram-tabular_d1 - trim={18pt 15pt 18pt 15pt},clip,
      \includegraphics[height=2.5cm]{../fig/diagram-tab1}%2.75
      \caption{Transformation from the original domain into the interpretable representation \(\mathcal{X} \rightarrow \mathcal{X}^\star\).\label{fig:tab:1}}% prime
  \end{subfigure}
  \par\bigskip % force a bit of vertical whitespace
  \begin{subfigure}[t]{.99\textwidth}
      \centering
      % diagram-tabular_d2 - trim={18pt 15pt 18pt 15pt},clip,
      \includegraphics[height=2.5cm]{../fig/diagram-tab2}%2.75
      \caption{Transformation from the interpretable representation into the original domain \(\mathcal{X}^\star \rightarrow \mathcal{X}\).\label{fig:tab:2}}% prime
  \end{subfigure}
  \caption{%
% Example of interpretable representation transformation in both directions for tabular data. %
Depiction of a forward and backward transformation between the original and interpretable representations of tabular data. %
Panel~(\subref{fig:tab:1}) shows the discretisation and binarisation steps required to represent a data point as a binary on/off vector; Panel~(\subref{fig:tab:2}) illustrates this procedure in the opposite direction. % depicts
The forward transformation is \emph{deterministic} given a fixed discretisation algorithm (i.e., binning of numerical features), however moving from the IR to the original domain is \emph{stochastic} since it requires random sampling.% non-deterministic
\label{fig:tab}}% Non-bijective transformation in tabular data domain. % domain % whereas % the inverse of %, showing its \emph{non-bijectiveness} % discrete vector % procedure % in the reverse process
\end{figure}

\begin{figure}[t]%bh
    \centering
    \begin{subfigure}[t]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{../fig/tabular_example.pdf}% tab_1.pdf
        \caption{Discretised and binarised numerical features become the components of the tabular interpretable representation (\(x^\star\)).\label{fig:tab_ex:1}}
    \end{subfigure}
    %\hfill
    \hspace{0.033333333\linewidth}
    \begin{subfigure}[t]{0.45\textwidth}
        \centering
        \includegraphics[width=0.888888889\textwidth]{../fig/tabular_explanation.pdf}% fig/tab_2.pdf % tab_2_big.pdf
        \caption{Explanation shows influence of IR components on predicting the \emph{grey} class for the red \(\star\) instance and, more generally, the entire \(x^\star = (1, 1)\) hyper-rectangle.\label{fig:tab_ex:2}}% instance marked with 
    \end{subfigure}
    \caption{%
Example of an influence-based explanation of tabular data with the interpretable representation built upon \emph{discretisation} (\(x^\prime\)) and \emph{binarisation} (\(x^\star\)) of numerical features. % importance
Panel~(\subref{fig:tab_ex:1}) illustrates an instance (red \(\star\)) to be explained, which is being predicted by a black-box model. %
The dashed blue lines mark binning of numerical attributes, grey and green dots denote two classes, \(x^\prime\) is the (intermediate) discrete representation, and \(x^\star\) encodes the binary IR created for the \(\star\) data point. %
Panel~(\subref{fig:tab_ex:2}) depicts the magnitude of the influence that \(x_1^\star: 75 \leq x_1\) and \(x_2^\star: 40 < x_2 \leq 80\) have on predicting the \emph{grey} class for the \(\star\) instance (and more broadly any other data point located within the same hyper-rectangle).% feature % as well as
\label{fig:tab_ex}}% limited to 2-D for the benefit of visualisation % classified
\end{figure}

(text:meta-explainers:surrogates:image:data-sampling)=
## Data Sampling ##

(text:meta-explainers:surrogates:image:explanation-generation)=
## Explanation Generation ##
