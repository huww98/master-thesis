\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{scutmaster}[SCUT Master Thesis Template]

\LoadClass[
    a4paper,openany,
    scheme=chinese,
    zihao=-4,
    linespread=1.5
]{ctexbook}

\RequirePackage[
    left=25mm,right=25mm,bottom=25mm,top=25mm,
    headheight=15mm,
    footskip=10mm
]{geometry}

\ctexset{
    chapter={
        format=\zihao{-2}\heiti\centering,
        beforeskip=0.5em plus 1.0em minus 0.1em,
        afterskip=0.5em plus 0.5em minus 0.1em,
    },
    section={
        format=\zihao{-3}\heiti,
        beforeskip=0.5em plus 1.0em minus 0.1em,
        afterskip=0.5em plus 0.5em minus 0.1em,
    },
    subsection={
        format=\zihao{4}\heiti,
        beforeskip=0.5em plus 1.0em minus 0.1em,
        afterskip=0.5em plus 0.5em minus 0.1em,
    },
    subsubsection={
        format=\zihao{-4}\heiti,
        beforeskip=0.5em plus 1.0em minus 0.1em,
        afterskip=0.5em plus 0.5em minus 0.1em,
    },
}

\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

\floatname{algorithm}{算法}
\renewcommand{\listalgorithmname}{算法}

\RequirePackage{amssymb}
\RequirePackage{amsmath}

\renewcommand{\theequation}{\arabic{chapter}-\arabic{equation}}
\renewcommand{\thetable}{\arabic{chapter}-\arabic{table}}
\renewcommand{\thefigure}{\arabic{chapter}-\arabic{figure}}

\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancypagestyle{main}{
    \fancyhead[EC]{\zihao{5}\leftmark}
    \fancyhead[OC]{\zihao{5}华南理工大学硕士论文}
    \fancyhead[L,R]{}
    \renewcommand\headrulewidth{1.5pt}
    \fancyfoot[C]{\zihao{5}\thepage}
}

\newcommand{\keyword}[1]{
    \paragraph{#1}
}

\clubpenalty=2000
\widowpenalty=3000
