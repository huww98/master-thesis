\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{scutmaster}[SCUT Master Thesis Template]

\gdef\@blind{}
\DeclareOption{blind}{\gdef\@blind{true}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessOptions\relax

\newcommand{\hideinblind}[1]{\ifx\@blind\@empty#1\fi}

\LoadClass[
    a4paper,openany,
    scheme=chinese,
    zihao=-4,
    fontset=none,
    linespread=1.625, % 对应Word中的1.5倍行距
]{ctexbook}

\xeCJKsetup{AutoFakeBold=true}
\RequirePackage[
    left=25mm,right=25mm,bottom=25mm,top=25mm,
    headheight=15mm,
    footskip=10mm
]{geometry}

\setmainfont{Times New Roman}

\ctexset{
    fontset=windows,
    chapter={
        format=\zihao{-2}\heiti\centering,
        beforeskip=1.0ex plus 1.0ex minus 0.1ex,
        afterskip=1.0ex plus 0.5ex minus 0.1ex,
    },
    section={
        format=\zihao{-3}\heiti,
        beforeskip=1.0ex plus 1.0ex minus 0.1ex,
        afterskip=1.0ex plus 0.5ex minus 0.1ex,
    },
    subsection={
        format=\zihao{4}\heiti,
        beforeskip=1.0ex plus 1.0ex minus 0.1ex,
        afterskip=1.0ex plus 0.5ex minus 0.1ex,
    },
    subsubsection={
        format=\zihao{-4}\heiti,
        beforeskip=1.0ex plus 1.0ex minus 0.1ex,
        afterskip=1.0ex plus 0.5ex minus 0.1ex,
    },
    paragraph={
        indent=\parindent,
        beforeskip=0.5ex plus 1ex minus 0.2ex,
    },
    contentsname={目\texorpdfstring{\quad}{}录},
    listfigurename={插图目录},
    listtablename={表格目录},
}
\setlength{\parskip}{0pt plus 2pt}

\RequirePackage[inline]{enumitem}
\setlist{
    itemsep=0pt plus 2pt,
    parsep =0pt plus 2pt,
    topsep =1pt plus 2pt minus .5pt,
}
\setlist[enumerate,1]{label=(\arabic*)}

\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

\floatname{algorithm}{算法}
\renewcommand{\listalgorithmname}{算法}

\RequirePackage{amssymb}
\RequirePackage{amsmath}

\renewcommand{\theequation}{\thechapter-\arabic{equation}}
\renewcommand{\thetable}{\thechapter-\arabic{table}}
\renewcommand{\thefigure}{\thechapter-\arabic{figure}}

\RequirePackage{caption}
\captionsetup{labelsep=space}

\RequirePackage{fancyhdr}
\pagestyle{fancy}
\fancypagestyle{main}{
    \fancyhead[OC]{\zihao{5}\leftmark}
    \fancyhead[EC]{\zihao{5}华南理工大学硕士学位论文}
    \fancyhead[L,R]{}
    \renewcommand\headrulewidth{1.5pt}
    \fancyfoot[C]{\zihao{5}\thepage}
}

\newcommand{\keyword}[1]{
    \vskip\baselineskip
    {\noindent\heiti\bf #1}%
}

\clubpenalty=2000
\widowpenalty=3000

\DeclareRobustCommand*{\titleEN}[1]{\gdef\@titleEN{#1}}
\DeclareRobustCommand*{\authorEN}[1]{\gdef\@authorEN{#1}}
\DeclareRobustCommand*{\classificationnumber}[1]{\gdef\@classificationnumber{#1}}
\DeclareRobustCommand*{\studentnumber}[1]{\gdef\@studentnumber{#1}}
\DeclareRobustCommand*{\phone}[1]{\gdef\@phone{#1}}
\DeclareRobustCommand*{\email}[1]{\gdef\@email{#1}}
\DeclareRobustCommand*{\address}[1]{\gdef\@address{#1}}

\DeclareRobustCommand*{\degree}[2]{\gdef\@degreeclass{#1}\gdef\@degree{#2}}
\DeclareRobustCommand*{\major}[2]{\gdef\@major{#1}\gdef\@majordir{#2}}

\DeclareRobustCommand*{\supervisor}[2]{\gdef\@supervisor{#1}\gdef\@supervisortitle{#2}}
\DeclareRobustCommand*{\supervisorEN}[2]{\gdef\@supervisortitleEN{#1}\gdef\@supervisorEN{#2}}
\DeclareRobustCommand*{\school}[1]{\gdef\@school{#1}}

\RequirePackage{tabularx}

\newcommand{\dateform}{\hphantom{2023}年\hphantom{12}月\hphantom{31}日}

\newenvironment{standalonepage}{
    \cleardoublepage
    \pagestyle{empty}
}{\newpage}

\RequirePackage{ulem}

\renewcommand{\maketitle}{
    \pagestyle{empty}
    \pagenumbering{Roman}
    \begin{titlepage}
        \null\vfil
        \begin{center}
            \heiti
            \includegraphics[width=0.8\textwidth]{figures/SCUT}\par
            \zihao{-0}专业学位硕士学位论文\par\vfill
            \zihao{2}\uline{\@title}\par\vfill

            \zihao{3}
            \hbadness=10000 % suppress warning
            \begin{tabular}{W{s}{6.5em} W{c}{12em}}
                \hideinblind{
                    作者姓名 & \@author\\
                    \cline{2-2}
                }
                学位类别 & \@degreeclass\\
                \cline{2-2}
                \hideinblind{
                    指导教师 & \@supervisor~\@supervisortitle\\
                    \cline{2-2}
                            &\\
                    \cline{2-2}
                }
                所在学院 & \@school\\
                \cline{2-2}
                论文提交日期 & \@date\\
                \cline{2-2}
            \end{tabular}
        \end{center}
        \vfil\null
    \end{titlepage}
}
\newcommand{\maketitleEN}{
    \begin{standalonepage}
        \null\vfil
        \vskip 60pt
        \begin{center}
            \Large\textbf{\@titleEN}\par\vskip 80pt
            \large A Dissertation Submitted for the Degree of Master
            \par\vskip 80pt
            \begin{tabular}[t]{l}
                \textbf{Candidate:} \@authorEN\\
                \textbf{Supervisor:} \@supervisortitleEN~\@supervisorEN\\
            \end{tabular}
            \par\vskip 160pt
            South China University of Technology \\
            Guangzhou, China
        \end{center}
        \vfil\null
    \end{standalonepage}
}
\newcommand{\nominationpage}{
    \begin{standalonepage}
        \ctexset{linestretch=10} % suppress warning
        \heiti\zihao{4}\noindent
        \begin{tabularx}{\textwidth}{w{s}{3em}@{：}Xl}
            分类号 & \@classificationnumber & 学校代码：10561 \\
            学号  & \@studentnumber &  \\
        \end{tabularx}
        \vskip 80pt
        \zihao{-2}
        \begin{center}
            华南理工大学硕士学位论文

            \vskip 50pt\zihao{-1}
            \textbf{\@title}\par
        \end{center}

        \vfill
        \setlength\extrarowheight{1em}
        \songti\zihao{5}\noindent
        \begin{tabularx}{\textwidth}{XX}
            作者姓名：\@author     & 指导教师姓名、职称：\@supervisor~\@supervisortitle\\
            申请学位级别：\@degree & 学科专业名称：\@major \\
            研究方向：\@majordir  & \\
            论文提交日期：\@date      & 论文答辩日期：\dateform\\
            学位授予单位：华南理工大学 & 学位授予日期：\dateform\\
            答辩委员会成员： & \\
            主席：\underline{\hspace{10em}}& \\
            \multicolumn{2}{l}{
                委员：\underline{\hspace{30em}}
            } \\
        \end{tabularx}
    \end{standalonepage}
}

\newcommand{\declareoforiginality}{
    \begin{standalonepage}
    \linespread{1.2}
    \songti\zihao{4}
    \begin{center}
        \bfseries\zihao{2}
        华南理工大学\\
        学位论文原创性声明
    \end{center}
    \vskip\baselineskip

    本人郑重声明：所呈交的论文是本人在导师的指导下独立进行研究所取得的研究成果。除了文中特别加以标注引用的内容外，本论文不包含任何其他个人或集体已经发表或撰写的成果作品。对本文的研究做出重要贡献的个人和集体，均已在文中以明确方式标明。本人完全意识到本声明的法律后果由本人承担。

    \vskip\baselineskip\noindent
    作者签名：\hspace{10em} 日期：\dateform

    \vfill
    \begin{center}
        \bfseries\zihao{2}
        学位论文版权使用授权书
    \end{center}
    \vskip 2ex

    本学位论文作者完全了解学校有关保留、使用学位论文的规定，即：研究生在校攻读学位期间论文工作的知识产权单位属华南理工大学。学校有权保存并向国家有关部门或机构送交论文的复印件和电子版，允许学位论文被查阅（除在保密期内的保密论文外）；学校可以公布学位论文的全部或部分内容，可以允许采用影印、缩印或其它复制手段保存、汇编学位论文。本人电子文档的内容和纸质论文的内容相一致。

    本学位论文属于：

    \begin{itemize}
        \setlength\itemsep{0pt}
        \item[$\square$]保密（校保密委员会审定为涉密学位论文时间：\dateform ），于\dateform 解密后适用本授权书。
        \item[\rlap{$\checkmark$}$\square$]不保密,同意在校园网上发布，供校内师生和与学校有共享协议的单位浏览；同意将本人学位论文编入有关数据库进行检索，传播学位论文的全部或部分内容。
    \end{itemize}

    (请在以上相应方框内打“$\checkmark$”)

    \vskip\baselineskip\noindent%
    \begin{tabular}{@{}ll@{}}
    作者签名：    & 日期：\dateform\\
    指导教师签名：& 日期：\dateform\\
    作者联系电话：\@phone & 电子邮箱：\@email\\
    \multicolumn{2}{@{}p{\linewidth}@{}}{
        联系地址(含邮编)：\@address
    }
    \end{tabular}
    \end{standalonepage}
}

\let\bookfrontmatter\frontmatter
\renewcommand{\frontmatter}{
    \bookfrontmatter
    \pagestyle{plain}
}

\let\bookmainmatter\mainmatter
\renewcommand{\mainmatter}{
    \bookmainmatter
    \pagestyle{main}
}

\RequirePackage{gbt7714}
\bibliographystyle{gbt7714-numerical}
\setlength\bibsep{1pt plus 1pt minus .5pt}

\RequirePackage{tocbibind}

\AtBeginDocument{
    \@ifpackageloaded{hyperref}{
        \hypersetup{pdftitle={\@title}}
        \hideinblind{
            \hypersetup{pdfauthor={\@author}}
        }
    }{}
}
