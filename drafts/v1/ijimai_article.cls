%% !TEX root = plantilla.tex
%
%
% First version: 2020/05/15
% Last updated: 
%
% Created: Daniel Pérez-Palau
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ijimai_article}

\LoadClass[twocolumn,a4paper]{article}


%
%---paquetes---------------------
%
%idioma i codificación
%\RequirePackage{babel}
%\RequirePackage[utf8]{inputenc} %descomentar esta línea si se trabaja en linux
%\RequirePackage[latin1]{inputenc} %comentar esta línea si se trabaja en linux

%
%%simbolos
%\RequirePackage{amsmath}

%formato de página
%\RequirePackage{fullpage}
%\RequirePackage[margin=8in]{geometry}

%\RequirePackage{newtxtext}
%\RequirePackage{newtxmath}

\RequirePackage{iftex}

%combine iflualatex and ifxetex from https://tex.stackexchange.com/questions/47576/combining-ifxetex-and-ifluatex-with-the-logical-or-operation
\newif\ifxetexorluatex
\ifxetex
  \xetexorluatextrue
\else
  \ifluatex
    \xetexorluatextrue
  \else
    \xetexorluatexfalse
  \fi
\fi

\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{mathabx}
\RequirePackage{amsfonts}

%fonts and letters
\RequirePackage[9pt]{extsizes}


\ifxetexorluatex
	\RequirePackage{fontspec}
	\setmainfont{LibertinusSerif}[ 
	%Path = /Users/daniel/Documents/Research_projects/IJIMAI/latex/font/serif/,
	Path = ./font/serif/,
	Extension = .otf,
	UprightFont = *-Regular,
	BoldFont = *-Bold,
	ItalicFont = *-Italic,
	BoldItalicFont = *-BoldItalic
	]
	
	
	\setsansfont{LibertinusSans}[ 
	%Path = /Users/daniel/Documents/Research_projects/IJIMAI/latex/font/serif/,
	Path = ./font/other/,
	Extension = .otf,
	UprightFont = *-Regular,
	BoldFont = *-Bold,
	ItalicFont = *-Italic
	]
	
	\setsansfont{LibertinusMono}[ 
	%Path = /Users/daniel/Documents/Research_projects/IJIMAI/latex/font/serif/,
	Path = ./font/other/,
	Extension = .otf,
	UprightFont = *-Regular,
	]
\else
	\ClassWarningNoLine{ijimai_article}{The fonts are not loaded properly. The output can differ from the final version of the paper. Run the code with LuaLatex option to get the same font letter output.}
	\usepackage[T1]{fontenc}
	\usepackage{tgpagella}
\fi

\RequirePackage{titling}
\RequirePackage{indentfirst}
%
%%imagenes
%\RequirePackage[dvips]{epsfig}
\RequirePackage{graphicx}
\RequirePackage{color}
%
%colores en tablas
\RequirePackage{colortbl}


%listas
\RequirePackage{enumitem}
\setlist{nosep, leftmargin=11pt}


%figures and tables
\RequirePackage[figurename=Fig.]{caption}
\DeclareCaptionLabelFormat{ijimai_format}{#1#2.~}
\DeclareCaptionLabelSeparator{null}{}
\renewcommand{\figurename}{Fig. }
\captionsetup[figure]{labelformat=ijimai_format,labelsep=null}

\RequirePackage{tabularx}

\RequirePackage{lettrine}
\RequirePackage{stringstrings}
\newcommand{\fancyFirstWord}[1]{\lettrine[lines=2]{\textcolor{azulijimai}{\substring{#1}{1}{1}}}{\textcolor{azulijimai}{\substring{#1}{2}{$}}}}%$





\RequirePackage[left=2cm,right=2cm, top=2cm, bottom=2cm]{geometry}

\RequirePackage{xcolor}
\RequirePackage{sectsty}
\RequirePackage{etoolbox}
%\RequirePackage{titlesec}
\RequirePackage[explicit]{titlesec}

%creamos secciones y subsecciones:
\renewcommand\thesection{{\color{azulijimai}\Roman{section}}}
\renewcommand\thesubsection{{\color{azulijimai}\Alph{subsection}}}
\renewcommand\thesubsubsection{{\color{azulijimai}\arabic{subsubsection}}}
\sectionfont{\color{azulijimai}}

\titleformat{\section}
	{\color{azulijimai}\normalfont\center\large\scshape}
	{\thesection.}
	{1em}
	{#1}
	[{\titlerule[0.8pt]}]

\subsectionfont{\color{azulijimai}}
\titleformat{\subsection}
	{\color{azulijimai}\normalfont\em}
	{\thesubsection. }
	{0pt}
	{#1}


\subsubsectionfont{\color{azulijimai}}
\titleformat{\subsubsection}
	{\color{azulijimai}\normalfont\em}
	{\thesubsubsection. }
	{0pt}
	{#1}

%Change the appendix format
\newcommand\appenAcknow{
\appendix
\renewcommand\thesection{{\color{azulijimai}}}
\titleformat{\section}
	{\color{azulijimai}\normalfont\center\large\scshape}
	{\thesection}
	{1em}
	{##1}
	[{\titlerule[0.8pt]}]
}



%\definecolor{azulijimai}{HTML}{0097cd}
\definecolor{azulijimai}{HTML}{008DC6}
\definecolor{azulfondo}{HTML}{E5F3F9}
%
\newcommand{\director}[1]{\gdef\@director{#1}}
\renewcommand{\abstract}[1]{\gdef\@theabstract{#1}}
\newcommand{\keywords}[1]{\gdef\@thekeywords{#1}}
\newcommand{\affiliation}[1]{\gdef\@theaffiliation{#1}}

\newcommand{\mailinfo}[1]{\gdef\@themailinfo{#1}}

\keywords{}
\abstract{}
\affiliation{}
\mailinfo{}






%creamos el nuevo título
\renewcommand{\maketitle}{
\twocolumn[
{
\begin{@twocolumnfalse}
\begingroup  
\begin{tabular}{ll}
\begin{tabular}{p{0.7\textwidth}}
	{\Huge  {\thetitle}  } \\\vspace{0.3cm}
	{\color{azulijimai}\theauthor }\\\vspace{0.3cm}
	\@theaffiliation \\\vspace{0.3cm}
	\begin{tabular}{p{5cm}}
		\arrayrulecolor{azulijimai}\hline
	\end{tabular}\\
	{\color{azulijimai}\MakeUppercase{\bf Abstract}}\\\arrayrulecolor{azulijimai}\hline
	\@theabstract
\end{tabular}&
\begin{tabular}{p{0.2\textwidth}}
\includegraphics[width= 0.2\textwidth]{logo_unirv}\\
{\color{azulijimai}\MakeUppercase{\bf Keywords}}\\\arrayrulecolor{azulijimai}\hline
\@thekeywords\\
\end{tabular}\\

\end{tabular}
\vspace{1cm}
\endgroup
\end{@twocolumnfalse}
}
]
\makemails
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make two kind of footnotes, one for authors other of usual footnotes.
%idea extracted (and adapted) from user 31729 and Ethan Bolker (https://tex.stackexchange.com/questions/312793/toggle-presence-of-footnote-rule)
%%%%\newcommand{\makemails}{}

\RequirePackage{footmisc}

\newcommand{\makemails}[1]{%
\let\xxxx\thempfn
\let\thempfn\relax% Remove footnote number printing mechanism
\footnotetext[0]{%
\vspace{0.1in}%
\hspace{-0.6cm}
\fcolorbox{white}{azulfondo}{\parbox{0.96\columnwidth}{\textcolor{azulijimai}{* Corresponding authors:}\\ \@themailinfo}}}% Print footnote text
\let\thempfn\xxxx% Restore footnote number printing mechanism
}

\RequirePackage{bophook}

\newif\iffootnoterule

\AtBeginDocument{%
\let\latex@@footnoterule\footnoterule

\renewcommand\footnoterule{%
  \iffootnoterule
  \latex@@footnoterule%
  \else
  \advance\skip\footins 4\p@\@plus2\p@\relax%
  \fi
}
}

\let\realfootnote\footnote

\renewcommand{\footnote}[1]{%
\global\footnoteruletrue
\realfootnote{#1}
}

\AtBeginPage{%
\global\footnoterulefalse
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{wrapfig}
\newcommand{\authorbio}[3]{
\begin{wrapfigure}{l}{1in}

\vspace{-0.6cm}
\centering
\includegraphics[width=1in, height=1.25in]{#1}
\end{wrapfigure}

\noindent\textcolor{azulijimai}{\bf \Large#2}

\noindent#3
}

\RequirePackage{cite}
\renewcommand\citedash{\hbox{]--[}\penalty\citepunctpenalty} 
\renewcommand\citepunct{], \hskip.13emplus.1emminus.1em\relax[\penalty\citepunctpenalty %
}
\RequirePackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=black,      
    urlcolor=cyan,
}




%avoid hyphen end of line
\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000