
def summarize_gsea(gsea_input, client,n_max=5):
    content=f"""# Gene Program Summarizer
## Task
Analyze gene set enrichment results and the associated gene list to generate a concise summary that captures the core biological function or identity of the gene program.

## Input
1. Enrichment results from pathway databases (GO, Reactome, PANTHER)
   - Term names with statistical significance metrics (p-value/FDR)
   - Gene counts or percentages per term
   
2. List of genes in the program (gene symbols)

## Output
1. Provide a 1-10 word summary that represents:
   - The fundamental biological processes/functions
   - The dominant cellular components or contexts
   - Key pathways or molecular activity

2. Extract up to {n_max} genes that are most relevant to the identified functions. 
3. For each important point, describe your reasoning and supporting information referencing genes identified in step 2.

Provide these 3 elements separated by a | symbol. 

## Analysis Strategy
1. **Identify the strongest signals**
   - Focus on terms with lowest p-values/FDR and highest fold enrichment
   - Prioritize terms with higher gene counts
   
2. **Find convergent themes**
   - Look for recurring concepts across different databases
   - Identify the highest-level biological concept that still maintains specificity
   
3. **Determine functional hierarchy**
   - Distinguish between primary functions and supporting mechanisms
   - Prioritize biological processes over molecular components when appropriate
   
4. **Extract key modifiers**
   - Identify tissue-specific or condition-specific aspects if strongly evident
   - Note developmental stage or cellular state if consistently represented

## Example Summaries

| Sample Enriched Terms | 1-10 Word Summary | Key Genes | Reasoning |
|----------------------|-----------------|-----------|----------|
| Inflammatory response, Cytokine signaling, NF-kB activation, Leukocyte migration | "Inflammatory Response" | IL6, TNF, NFKB1, IL1B, CXCL8 | Strong enrichment in cytokine signaling pathways; IL6 and TNF are master regulators of inflammation; NFKB1 is a central transcription factor in this process |
| DNA repair, Double-strand break repair, Homologous recombination, BRCA1 complex | "DNA Double-Strand Break Repair" | BRCA1, RAD51, ATM, PARP1, XRCC5 | All pathways converge on repair mechanisms; BRCA1 and RAD51 are essential for homologous recombination; ATM initiates repair signaling |
| Neuronal differentiation, Axon guidance, Synaptogenesis, Neurite outgrowth | "Neuronal Development and Synapse Formation" | BDNF, NTRK2, DLG4, ROBO1, NCAM1 | Terms indicate developmental processes; BDNF and NTRK2 regulate neuronal growth; DLG4 is critical for synapse organization |
| Oxidative phosphorylation, Mitochondrial respiration, Electron transport chain | "Mitochondrial Energy Production" | ATP5F1A, NDUFV1, UQCRC1, COX4I1, SDHA | All terms relate to cellular respiration; these genes encode components of different respiratory chain complexes |

## Guidelines for Clarity
- Use standard terminology from the field when possible
- Acceptable to use common acronyms (EMT, OXPHOS, etc.) if they are clearer
- Prioritize functional descriptions over participant lists
- For highly specific gene sets, maintain precision over generality

## Complete Output Example
"Inflammatory Response" | IL6, TNF, NFKB1, IL1B, CXCL8 | The gene set shows strong enrichment in inflammatory pathways (FDR < 0.001). The presence of multiple cytokines (IL6, IL1B) and chemokines (CXCL8) indicates a robust inflammatory response program. NFKB1 acts as a master regulator coordinating this response, while TNF signaling reinforces the inflammatory cascade. These genes collectively participate in immune cell recruitment and activation processes.
""" + f"here is the input: {gsea_input}"

    response =client.models.generate_content(
        model="gemini-2.0-flash", contents=content
    )
    return response.text
