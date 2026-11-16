from rdflib import Graph, Literal, RDF, RDFS, OWL, Namespace, URIRef

# 1. Setup Namespaces
PPB = Namespace("http://peachtreepantry.com/ontology/")
g = Graph()
g.bind("ppb", PPB)
g.bind("owl", OWL)

# 2. Define Core Classes (Nodes)
# These represent the primary entities in the HR domain
classes = [
    ("Employee", "A person working at Peachtree Pantry & Beverage Co."),
    ("JobRole", "A specific professional position (e.g., Hotline Cook)."),
    ("Department", "An organizational unit (e.g., Retail Operations)."),
    ("Location", "A physical store or regional office."),
    ("Region", "A geographical state entity (GA, NY, VA)."),
    ("Policy", "A normative document governing employee conduct."),
    ("SOP", "A standard operating procedure for a specific workflow."),
    ("ProcessStep", "An individual step within a multi-stage workflow."),
    ("Form", "A required document for a process (e.g., Form 102-B)."),
    ("DocumentChunk", "A specific snippet of text from the source corpus.")
]

for cls_name, comment in classes:
    cls_uri = PPB[cls_name]
    g.add((cls_uri, RDF.type, OWL.Class))
    g.add((cls_uri, RDFS.comment, Literal(comment)))

# 3. Define Object Properties (Relationships/Edges)
# These enable regional overrides and sequential logic
properties = [
    ("appliesIn", PPB.Policy, PPB.Region, "Links a policy to the region it governs."),
    ("reportsTo", PPB.Employee, PPB.Employee, "Hierarchical reporting line."),
    ("hasRole", PPB.Employee, PPB.JobRole, "The role assigned to an employee."),
    ("requiresStep", PPB.SOP, PPB.ProcessStep, "Connects a process to its sequential steps."),
    ("nextStep", PPB.ProcessStep, PPB.ProcessStep, "The sequence logic between steps."),
    ("requiresForm", PPB.ProcessStep, PPB.Form, "A form needed to complete a step."),
    ("locatedAt", PPB.Employee, PPB.Location, "The physical site where an employee works."),
    ("inRegion", PPB.Location, PPB.Region, "Links a specific store to a state region."),
    ("mentionedIn", URIRef("http://www.w3.org/2002/07/owl#Thing"), PPB.DocumentChunk, "Traceability to source text.")  
]

for prop_name, domain, range_cls, comment in properties:
    prop_uri = PPB[prop_name]
    g.add((prop_uri, RDF.type, OWL.ObjectProperty))
    g.add((prop_uri, RDFS.domain, domain))
    g.add((prop_uri, RDFS.range, range_cls))
    g.add((prop_uri, RDFS.comment, Literal(comment)))

# 4. Define Data Properties (Attributes)
# Captures metadata like effective dates and accrual rates
data_properties = [
    ("effectiveDate", PPB.Policy, "The date a policy became active."),
    ("accrualRate", PPB.Policy, "The mathematical rate of leave accrual."),
    ("employeeID", PPB.Employee, "Unique ID from HRIS."),
    ("provenanceID", PPB.DocumentChunk, "The source Document ID (e.g., PPB-HB-2026-V1).")
]

for prop_name, domain, comment in data_properties:
    prop_uri = PPB[prop_name]
    g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
    g.add((prop_uri, RDFS.domain, domain))
    g.add((prop_uri, RDFS.comment, Literal(comment)))

# 5. Serialize and Save
g.serialize(destination="data/ppb_hr_ontology.owl", format="xml")
print("[+] Ontology generated successfully: data/ppb_hr_ontology.owl")
