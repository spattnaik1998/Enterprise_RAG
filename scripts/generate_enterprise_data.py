#!/usr/bin/env python3
"""
Enterprise MSP Synthetic Data Generator
=========================================
Generates five interconnected datasets simulating the back-office systems
of TechVault MSP, a fictional Managed Service Provider with 50 enterprise clients.

Source systems modelled:
  1. Billing System (QuickBooks Enterprise) -- ~700 invoice records
  2. PSA System (ConnectWise Manage)        -- ~300 service ticket records
  3. CRM System (HubSpot CRM)              -- 50 client profile records
  4. Communications Log (Exchange Online)   -- ~250 email/reminder records
  5. Contracts Repository (SharePoint)      -- 50 service agreement records

Run:
    python scripts/generate_enterprise_data.py

Output: data/enterprise/*.json
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

fake = Faker("en_US")
Faker.seed(42)
random.seed(42)

TODAY = date(2026, 2, 28)
BILLING_START = date(2025, 1, 1)   # 13 months of billing history
BILLING_END   = date(2026, 1, 1)   # last monthly billing run
TAX_RATE      = 0.09
LATE_FEE_RATE = 0.015              # 1.5 % per month after due date

# ── MSP Profile ───────────────────────────────────────────────────────────────

MSP = {
    "name": "TechVault MSP",
    "tagline": "Managed IT Solutions for the Modern Enterprise",
    "address": "500 Innovation Drive, Suite 200, San Jose, CA 95110",
    "billing_email": "billing@techvaultmsp.com",
    "support_email": "support@techvaultmsp.com",
    "phone": "1-800-832-4885",
    "website": "www.techvaultmsp.com",
    "federal_tax_id": "47-8823164",
    "remittance": "First National Commercial Bank | ABA 121000248 | Acct 880042671",
    "payment_methods": ["ACH (preferred)", "Check payable to TechVault MSP",
                        "Wire Transfer", "Credit Card (3% surcharge)"],
}

# ── Service Catalogue ─────────────────────────────────────────────────────────

SERVICES = {
    "MSP_T1":      {"name": "Managed IT Support - Tier 1 (up to 25 users)",            "type": "recurring",    "unit": "month", "price": 1_500.00},
    "MSP_T2":      {"name": "Managed IT Support - Tier 2 (26-75 users)",               "type": "recurring",    "unit": "month", "price": 2_500.00},
    "MSP_T3":      {"name": "Managed IT Support - Tier 3 (76+ users)",                 "type": "recurring",    "unit": "month", "price": 4_500.00},
    "M365":        {"name": "Microsoft 365 Administration",                             "type": "per_user",     "unit": "user",  "price": 8.00},
    "SEC_STD":     {"name": "Cybersecurity Monitoring (EDR/Endpoint Protection)",       "type": "recurring",    "unit": "month", "price": 995.00},
    "SEC_ADV":     {"name": "Advanced Security Operations Center (SIEM/SOC)",           "type": "recurring",    "unit": "month", "price": 2_200.00},
    "CLOUD":       {"name": "Cloud Infrastructure Management (Azure/AWS)",              "type": "recurring",    "unit": "month", "price": 1_200.00},
    "NETWORK":     {"name": "Network Infrastructure Support & Monitoring",              "type": "recurring",    "unit": "month", "price": 750.00},
    "BCP":         {"name": "Business Continuity & Disaster Recovery",                  "type": "recurring",    "unit": "month", "price": 550.00},
    "VOIP":        {"name": "VoIP Phone System Support & Management",                   "type": "recurring",    "unit": "month", "price": 350.00},
    "COMPLY":      {"name": "IT Compliance Consulting (HIPAA/SOC2/PCI-DSS)",            "type": "flat_monthly", "unit": "month", "price": 740.00},
    "EMER":        {"name": "Emergency On-Site Support",                                "type": "hourly",       "unit": "hour",  "price": 225.00},
    "REMOTE_TM":   {"name": "Remote Support - Time & Materials",                        "type": "hourly",       "unit": "hour",  "price": 150.00},
    "PROJ_SEC":    {"name": "Security Assessment & Penetration Testing",                "type": "project",      "unit": "flat",  "price": 4_500.00},
    "PROJ_INFRA":  {"name": "Infrastructure Upgrade & Technology Refresh Project",      "type": "project",      "unit": "flat",  "price": 8_500.00},
    "PROJ_MIGRATE":{"name": "Cloud Migration & Modernisation Project",                  "type": "project",      "unit": "flat",  "price": 12_000.00},
    "PROJ_BACKUP": {"name": "Backup Infrastructure Overhaul",                           "type": "project",      "unit": "flat",  "price": 3_500.00},
    "PROJ_ASSESS": {"name": "IT Infrastructure Assessment & Technology Roadmap",        "type": "project",      "unit": "flat",  "price": 2_800.00},
    "PROJ_DR":     {"name": "Disaster Recovery Planning & Tabletop Exercise",           "type": "project",      "unit": "flat",  "price": 5_200.00},
}

# ── 50 Client Templates ───────────────────────────────────────────────────────
# reliability: EXCELLENT / GOOD / FAIR / POOR  -- drives invoice payment behaviour
# terms:       NET30 / NET45 / NET60

CLIENT_TEMPLATES = [
    # Healthcare (8)
    {"id":"CLT-001","name":"Lakewood Medical Group",              "industry":"Healthcare",           "size":"L","users": 95,"services":["MSP_T3","M365","SEC_ADV","CLOUD","BCP","COMPLY"],        "reliability":"FAIR",      "terms":"NET30","start":"2021-03-15","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-002","name":"Blue Ridge Dental Partners",          "industry":"Healthcare",           "size":"S","users": 18,"services":["MSP_T1","M365","SEC_STD","BCP"],                         "reliability":"GOOD",      "terms":"NET30","start":"2022-06-01","has_projects":False,"manager":"James Wei"},
    {"id":"CLT-003","name":"Northern Lights Healthcare System",   "industry":"Healthcare",           "size":"L","users":210,"services":["MSP_T3","M365","SEC_ADV","CLOUD","NETWORK","BCP","COMPLY"],"reliability":"POOR",    "terms":"NET45","start":"2020-11-01","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-004","name":"Summit Pediatric Associates",         "industry":"Healthcare",           "size":"S","users": 22,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"EXCELLENT", "terms":"NET30","start":"2023-01-15","has_projects":False,"manager":"James Wei"},
    {"id":"CLT-005","name":"Riverside Cardiology Center",         "industry":"Healthcare",           "size":"M","users": 45,"services":["MSP_T2","M365","SEC_STD","CLOUD","BCP"],                 "reliability":"GOOD",      "terms":"NET30","start":"2022-03-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-006","name":"Valley Vision Eye Care",              "industry":"Healthcare",           "size":"S","users": 12,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"EXCELLENT", "terms":"NET30","start":"2023-05-01","has_projects":False,"manager":"James Wei"},
    {"id":"CLT-007","name":"Crestwood Physical Therapy",         "industry":"Healthcare",           "size":"S","users": 20,"services":["MSP_T1","M365","SEC_STD","BCP"],                         "reliability":"FAIR",      "terms":"NET30","start":"2022-09-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-008","name":"Meridian Urgent Care Network",        "industry":"Healthcare",           "size":"M","users": 58,"services":["MSP_T2","M365","SEC_ADV","CLOUD","NETWORK"],             "reliability":"GOOD",      "terms":"NET30","start":"2021-07-15","has_projects":True, "manager":"Diana Reyes"},
    # Legal / Professional Services (7)
    {"id":"CLT-009","name":"Harrington & Associates LLP",         "industry":"Legal",                "size":"M","users": 38,"services":["MSP_T2","M365","SEC_STD","COMPLY"],                      "reliability":"EXCELLENT", "terms":"NET30","start":"2021-01-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-010","name":"Morrison Caldwell Legal Group",        "industry":"Legal",                "size":"M","users": 52,"services":["MSP_T2","M365","SEC_ADV","NETWORK","COMPLY"],             "reliability":"GOOD",      "terms":"NET30","start":"2020-08-01","has_projects":True, "manager":"Jason Cho"},
    {"id":"CLT-011","name":"Pacific Coast Accounting Partners",    "industry":"Professional Services","size":"S","users": 25,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"GOOD",      "terms":"NET45","start":"2023-03-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-012","name":"Apex Business Consulting",            "industry":"Professional Services","size":"S","users": 15,"services":["MSP_T1","M365","SEC_STD","VOIP"],                        "reliability":"FAIR",      "terms":"NET30","start":"2022-11-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-013","name":"Sterling Wealth Management",          "industry":"Finance",              "size":"S","users": 20,"services":["MSP_T1","M365","SEC_ADV","COMPLY"],                      "reliability":"EXCELLENT", "terms":"NET30","start":"2021-06-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-014","name":"Quantum Strategy Group",              "industry":"Professional Services","size":"S","users": 10,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"POOR",      "terms":"NET30","start":"2023-09-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-015","name":"Premier HR Solutions",                "industry":"Professional Services","size":"S","users": 18,"services":["MSP_T1","M365","SEC_STD","VOIP"],                        "reliability":"GOOD",      "terms":"NET30","start":"2022-04-01","has_projects":False,"manager":"Sarah Park"},
    # Finance / Insurance (6)
    {"id":"CLT-016","name":"Summit Financial Advisors",           "industry":"Finance",              "size":"M","users": 35,"services":["MSP_T2","M365","SEC_ADV","COMPLY","CLOUD"],              "reliability":"POOR",      "terms":"NET30","start":"2021-02-01","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-017","name":"Riverfront Insurance Agency",         "industry":"Insurance",            "size":"M","users": 42,"services":["MSP_T2","M365","SEC_STD","NETWORK"],                     "reliability":"FAIR",      "terms":"NET30","start":"2020-05-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-018","name":"Coastal Capital Partners",            "industry":"Finance",              "size":"S","users": 22,"services":["MSP_T1","M365","SEC_ADV","COMPLY"],                      "reliability":"EXCELLENT", "terms":"NET45","start":"2022-07-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-019","name":"Heritage Bank & Trust",               "industry":"Finance",              "size":"L","users":125,"services":["MSP_T3","M365","SEC_ADV","CLOUD","NETWORK","COMPLY"],    "reliability":"EXCELLENT", "terms":"NET45","start":"2020-01-15","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-020","name":"Pinnacle Asset Management",           "industry":"Finance",              "size":"M","users": 48,"services":["MSP_T2","M365","SEC_ADV","COMPLY"],                      "reliability":"GOOD",      "terms":"NET30","start":"2021-09-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-021","name":"Guardian Life Planning",              "industry":"Insurance",            "size":"S","users": 24,"services":["MSP_T1","M365","SEC_STD","BCP"],                         "reliability":"GOOD",      "terms":"NET30","start":"2022-12-01","has_projects":False,"manager":"Sarah Park"},
    # Manufacturing (6)
    {"id":"CLT-022","name":"Crossroads Manufacturing Inc.",       "industry":"Manufacturing",        "size":"L","users":145,"services":["MSP_T3","M365","SEC_STD","CLOUD","NETWORK","BCP"],       "reliability":"POOR",      "terms":"NET45","start":"2020-04-01","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-023","name":"Ironworks Fabrication Co.",           "industry":"Manufacturing",        "size":"M","users": 62,"services":["MSP_T2","M365","SEC_STD","NETWORK"],                     "reliability":"FAIR",      "terms":"NET30","start":"2021-10-01","has_projects":False,"manager":"Marcus Johnson"},
    {"id":"CLT-024","name":"Precision Components Ltd.",           "industry":"Manufacturing",        "size":"M","users": 78,"services":["MSP_T2","M365","SEC_STD","CLOUD","NETWORK"],             "reliability":"GOOD",      "terms":"NET30","start":"2022-01-15","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-025","name":"Atlas Industrial Systems",            "industry":"Manufacturing",        "size":"L","users":188,"services":["MSP_T3","M365","SEC_ADV","CLOUD","NETWORK","BCP"],       "reliability":"GOOD",      "terms":"NET45","start":"2020-07-01","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-026","name":"Keystone Plastics Group",             "industry":"Manufacturing",        "size":"M","users": 55,"services":["MSP_T2","M365","SEC_STD","NETWORK"],                     "reliability":"FAIR",      "terms":"NET30","start":"2021-11-01","has_projects":False,"manager":"Marcus Johnson"},
    {"id":"CLT-027","name":"Frontier Metal Works",                "industry":"Manufacturing",        "size":"S","users": 30,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"GOOD",      "terms":"NET30","start":"2023-02-01","has_projects":False,"manager":"Marcus Johnson"},
    # Construction / Real Estate (5)
    {"id":"CLT-028","name":"Metro Construction Co.",              "industry":"Construction",         "size":"L","users": 95,"services":["MSP_T3","M365","SEC_STD","CLOUD","VOIP"],                "reliability":"POOR",      "terms":"NET30","start":"2021-05-01","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-029","name":"Pacific Coast Realty Group",          "industry":"Real Estate",          "size":"M","users": 45,"services":["MSP_T2","M365","SEC_STD","VOIP"],                        "reliability":"FAIR",      "terms":"NET30","start":"2022-02-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-030","name":"Evergreen Property Management",       "industry":"Real Estate",          "size":"M","users": 38,"services":["MSP_T2","M365","SEC_STD","VOIP"],                        "reliability":"GOOD",      "terms":"NET45","start":"2021-08-01","has_projects":False,"manager":"Sarah Park"},
    {"id":"CLT-031","name":"Blackstone Development Group",        "industry":"Construction",         "size":"M","users": 55,"services":["MSP_T2","M365","SEC_STD","CLOUD"],                       "reliability":"FAIR",      "terms":"NET30","start":"2022-10-01","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-032","name":"Horizon Architecture & Design",       "industry":"Construction",         "size":"S","users": 28,"services":["MSP_T1","M365","SEC_STD","CLOUD"],                       "reliability":"GOOD",      "terms":"NET30","start":"2023-04-01","has_projects":False,"manager":"Sarah Park"},
    # Retail / Hospitality (6)
    {"id":"CLT-033","name":"Skyline Hospitality Group",           "industry":"Hospitality",          "size":"M","users": 68,"services":["MSP_T2","M365","SEC_STD","NETWORK","VOIP"],              "reliability":"FAIR",      "terms":"NET30","start":"2021-12-01","has_projects":True, "manager":"Linda Torres"},
    {"id":"CLT-034","name":"Pinnacle Retail Solutions",           "industry":"Retail",               "size":"M","users": 52,"services":["MSP_T2","M365","SEC_STD","CLOUD"],                       "reliability":"GOOD",      "terms":"NET30","start":"2022-03-15","has_projects":False,"manager":"Linda Torres"},
    {"id":"CLT-035","name":"Grand Vista Hotels",                  "industry":"Hospitality",          "size":"L","users":115,"services":["MSP_T3","M365","SEC_STD","CLOUD","NETWORK","VOIP"],      "reliability":"GOOD",      "terms":"NET30","start":"2020-09-01","has_projects":True, "manager":"Linda Torres"},
    {"id":"CLT-036","name":"Coastline Restaurant Group",          "industry":"Hospitality",          "size":"M","users": 42,"services":["MSP_T2","M365","SEC_STD","VOIP"],                        "reliability":"POOR",      "terms":"NET30","start":"2021-03-01","has_projects":False,"manager":"Linda Torres"},
    {"id":"CLT-037","name":"Aurora Luxury Spa & Wellness",        "industry":"Retail",               "size":"S","users": 22,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"GOOD",      "terms":"NET30","start":"2023-07-01","has_projects":False,"manager":"Linda Torres"},
    {"id":"CLT-038","name":"Marketbridge Commerce Solutions",     "industry":"Retail",               "size":"M","users": 58,"services":["MSP_T2","M365","SEC_STD","CLOUD"],                       "reliability":"FAIR",      "terms":"NET30","start":"2022-06-15","has_projects":True, "manager":"Linda Torres"},
    # Education (4)
    {"id":"CLT-039","name":"Oakdale Unified School District",     "industry":"Education",            "size":"L","users":280,"services":["MSP_T3","M365","SEC_STD","CLOUD","NETWORK"],             "reliability":"GOOD",      "terms":"NET45","start":"2020-08-15","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-040","name":"Westfield Community College",         "industry":"Education",            "size":"L","users":185,"services":["MSP_T3","M365","SEC_STD","CLOUD","NETWORK"],             "reliability":"EXCELLENT", "terms":"NET45","start":"2021-01-01","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-041","name":"Innovation Academy Charter School",   "industry":"Education",            "size":"M","users": 65,"services":["MSP_T2","M365","SEC_STD","CLOUD"],                       "reliability":"GOOD",      "terms":"NET30","start":"2022-08-15","has_projects":False,"manager":"Marcus Johnson"},
    {"id":"CLT-042","name":"Cypress Learning Institute",          "industry":"Education",            "size":"S","users": 32,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"FAIR",      "terms":"NET30","start":"2023-08-15","has_projects":False,"manager":"Marcus Johnson"},
    # Energy / Utilities (4)
    {"id":"CLT-043","name":"Desert Wind Energy",                  "industry":"Energy",               "size":"M","users": 48,"services":["MSP_T2","M365","SEC_STD","CLOUD","BCP"],                 "reliability":"EXCELLENT", "terms":"NET30","start":"2021-04-01","has_projects":True, "manager":"Jason Cho"},
    {"id":"CLT-044","name":"Pacific Solar Solutions",             "industry":"Energy",               "size":"M","users": 42,"services":["MSP_T2","M365","SEC_STD","CLOUD"],                       "reliability":"GOOD",      "terms":"NET30","start":"2022-05-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-045","name":"Cascade Power Management",            "industry":"Energy",               "size":"S","users": 28,"services":["MSP_T1","M365","SEC_STD","BCP"],                         "reliability":"GOOD",      "terms":"NET30","start":"2023-01-01","has_projects":False,"manager":"Jason Cho"},
    {"id":"CLT-046","name":"BlueSky Environmental Services",      "industry":"Energy",               "size":"S","users": 22,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"FAIR",      "terms":"NET45","start":"2022-09-15","has_projects":False,"manager":"Jason Cho"},
    # Logistics / Transportation (4)
    {"id":"CLT-047","name":"Harbor Logistics Solutions",          "industry":"Logistics",            "size":"M","users": 72,"services":["MSP_T2","M365","SEC_STD","CLOUD","NETWORK"],             "reliability":"FAIR",      "terms":"NET30","start":"2021-06-15","has_projects":True, "manager":"Marcus Johnson"},
    {"id":"CLT-048","name":"Rapid Transit Freight",               "industry":"Logistics",            "size":"M","users": 58,"services":["MSP_T2","M365","SEC_STD","NETWORK"],                     "reliability":"GOOD",      "terms":"NET30","start":"2022-01-01","has_projects":False,"manager":"Marcus Johnson"},
    {"id":"CLT-049","name":"Continental Supply Chain Partners",   "industry":"Logistics",            "size":"L","users":112,"services":["MSP_T3","M365","SEC_STD","CLOUD","NETWORK"],             "reliability":"GOOD",      "terms":"NET45","start":"2020-11-01","has_projects":True, "manager":"Diana Reyes"},
    {"id":"CLT-050","name":"Velocity Courier Network",            "industry":"Logistics",            "size":"S","users": 30,"services":["MSP_T1","M365","SEC_STD"],                               "reliability":"POOR",      "terms":"NET30","start":"2022-07-01","has_projects":False,"manager":"Marcus Johnson"},
]

# ── Helper utilities ──────────────────────────────────────────────────────────

def _add_month(d: date) -> date:
    m = d.month + 1
    y = d.year + (1 if m > 12 else 0)
    return d.replace(year=y, month=(m - 1) % 12 + 1, day=1)

def _terms_days(terms: str) -> int:
    return {"NET30": 30, "NET45": 45, "NET60": 60}.get(terms, 30)

def _invoice_status(due: date, reliability: str) -> str:
    """
    Determine payment status based on how overdue the invoice is and
    the client's historical payment reliability.

    Probability-of-payment table:
      Reliability  | <=30d | <=60d | <=90d | >90d
      EXCELLENT    |  99%  |  99%  |  100% | 100%
      GOOD         |  92%  |  97%  |  99%  |  99%
      FAIR         |  72%  |  82%  |  90%  |  94%
      POOR         |  40%  |  55%  |  68%  |  78%
    """
    if due > TODAY:
        return "OPEN"
    days = (TODAY - due).days
    if days <= 0:
        return "OPEN"

    p_table = {
        "EXCELLENT": [(30, .99), (60, .99), (90, 1.0),  (999, 1.0)],
        "GOOD":      [(30, .92), (60, .97), (90, .99),  (999, .99)],
        "FAIR":      [(30, .72), (60, .82), (90, .90),  (999, .94)],
        "POOR":      [(30, .40), (60, .55), (90, .68),  (999, .78)],
    }
    p_paid = next(p for thresh, p in p_table.get(reliability, p_table["GOOD"]) if days <= thresh)

    if random.random() < p_paid:
        return "PAID"
    if days > 90:
        return "OVERDUE_CRITICAL"
    if days > 60:
        return "OVERDUE"
    if days > 30:
        return "PAST_DUE"
    return "OPEN"

def _overdue_label(status: str, days: int) -> str:
    labels = {
        "OVERDUE_CRITICAL": f"OVERDUE -- CRITICAL ({days} days past due)",
        "OVERDUE":          f"OVERDUE ({days} days past due)",
        "PAST_DUE":         f"PAST DUE ({days} days past due)",
        "OPEN":             "OPEN -- within terms",
        "PAID":             "PAID",
    }
    return labels.get(status, status)

# ── 1. Client enrichment ──────────────────────────────────────────────────────

def enrich_clients(templates: list) -> list:
    clients = []
    for t in templates:
        state = fake.state_abbr()
        city  = fake.city()
        domain = t["name"].lower().replace(" ", "").replace("&", "and")[:20] + ".com"
        cfo   = fake.name()
        it_mgr = fake.name()
        ar_contact = fake.name()
        clients.append({
            **t,
            "domain": domain,
            "address": {
                "street": fake.street_address(),
                "city": city, "state": state,
                "zip": fake.zipcode(), "country": "US",
            },
            "contacts": {
                "cfo":        {"name": cfo,       "email": f"cfo@{domain}",      "phone": fake.phone_number()},
                "it_manager": {"name": it_mgr,    "email": f"it@{domain}",       "phone": fake.phone_number()},
                "ar_contact": {"name": ar_contact,"email": f"billing@{domain}",  "phone": fake.phone_number()},
            },
            "billing_contact_email": f"billing@{domain}",
        })
    return clients

# ── 2. Invoices ───────────────────────────────────────────────────────────────

def generate_invoices(clients: list) -> list:
    invoices = []
    inv_seq  = 1000

    for client in clients:
        # --- Monthly recurring invoices ---
        cur = BILLING_START
        while cur <= BILLING_END:
            inv_seq += 1
            inv_date = cur.replace(day=random.randint(1, 5))
            due_date = inv_date + timedelta(days=_terms_days(client["terms"]))

            line_items = []
            subtotal   = 0.0
            for svc_id in client["services"]:
                svc = SERVICES[svc_id]
                if svc["type"] == "project":
                    continue
                qty = client["users"] if svc["type"] == "per_user" else 1
                desc = f"{svc['name']} -- {cur.strftime('%B %Y')}"
                if svc_id == "EMER":       # emergency support: variable hours
                    qty = random.choice([0, 0, 1, 2, 3])
                    if qty == 0:
                        continue
                    desc = f"{svc['name']} -- {qty} hr(s) @ ${svc['price']:.2f}/hr"
                amount = round(qty * svc["price"], 2)
                line_items.append({
                    "service_code": svc_id, "description": desc,
                    "quantity": qty, "unit_price": svc["price"], "amount": amount,
                })
                subtotal += amount

            subtotal   = round(subtotal, 2)
            tax        = round(subtotal * TAX_RATE, 2)
            total      = round(subtotal + tax, 2)
            status     = _invoice_status(due_date, client["reliability"])
            days_out   = max(0, (TODAY - due_date).days) if status != "PAID" else 0
            paid_amt   = total if status == "PAID" else 0.0

            # Partial payment for PAST_DUE (sometimes client paid something)
            if status == "PAST_DUE" and random.random() < 0.25:
                paid_amt = round(total * random.uniform(0.3, 0.7), 2)

            invoices.append({
                "invoice_id":       f"INV-2025-{inv_seq:04d}",
                "client_id":        client["id"],
                "client_name":      client["name"],
                "billing_contact":  client["billing_contact_email"],
                "account_manager":  client["manager"],
                "billing_period":   cur.strftime("%B %Y"),
                "invoice_type":     "MONTHLY_RECURRING",
                "invoice_date":     inv_date.isoformat(),
                "due_date":         due_date.isoformat(),
                "payment_terms":    client["terms"],
                "status":           status,
                "days_outstanding": days_out,
                "line_items":       line_items,
                "subtotal":         subtotal,
                "tax_rate":         TAX_RATE,
                "tax_amount":       tax,
                "total_amount":     total,
                "amount_paid":      paid_amt,
                "balance_due":      round(total - paid_amt, 2),
                "currency":         "USD",
                "late_fee_applicable": status in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE"),
                "late_fee_amount":  round(total * LATE_FEE_RATE * max(days_out // 30, 1), 2)
                                    if status in ("OVERDUE", "OVERDUE_CRITICAL") else 0.0,
                "notes":            "Payment due per service agreement. "
                                    "Late payments subject to 1.5% monthly finance charge.",
                "source_system":    "QuickBooks Enterprise 2025",
                "source_system_type": "Billing",
            })
            cur = _add_month(cur)

        # --- Project invoices (for clients flagged has_projects) ---
        if client["has_projects"]:
            proj_svcs = [k for k, v in SERVICES.items() if v["type"] == "project"]
            for _ in range(random.randint(1, 3)):
                inv_seq += 1
                svc_id   = random.choice(proj_svcs)
                svc      = SERVICES[svc_id]
                # Random date within billing window
                days_offset  = random.randint(0, (BILLING_END - BILLING_START).days)
                inv_date = BILLING_START + timedelta(days=days_offset)
                due_date = inv_date + timedelta(days=30)
                amount   = round(svc["price"] * random.uniform(0.85, 1.40), 2)
                tax      = round(amount * TAX_RATE, 2)
                total    = round(amount + tax, 2)
                status   = _invoice_status(due_date, client["reliability"])
                days_out = max(0, (TODAY - due_date).days) if status != "PAID" else 0
                paid_amt = total if status == "PAID" else 0.0

                invoices.append({
                    "invoice_id":       f"INV-2025-{inv_seq:04d}",
                    "client_id":        client["id"],
                    "client_name":      client["name"],
                    "billing_contact":  client["billing_contact_email"],
                    "account_manager":  client["manager"],
                    "billing_period":   inv_date.strftime("%B %Y"),
                    "invoice_type":     "PROJECT",
                    "project_name":     svc["name"],
                    "invoice_date":     inv_date.isoformat(),
                    "due_date":         due_date.isoformat(),
                    "payment_terms":    "NET30",
                    "status":           status,
                    "days_outstanding": days_out,
                    "line_items":       [{"service_code": svc_id, "description": svc["name"],
                                          "quantity": 1, "unit_price": amount, "amount": amount}],
                    "subtotal":         amount,
                    "tax_rate":         TAX_RATE,
                    "tax_amount":       tax,
                    "total_amount":     total,
                    "amount_paid":      paid_amt,
                    "balance_due":      round(total - paid_amt, 2),
                    "currency":         "USD",
                    "late_fee_applicable": status in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE"),
                    "late_fee_amount":  round(total * LATE_FEE_RATE * max(days_out // 30, 1), 2)
                                        if status in ("OVERDUE", "OVERDUE_CRITICAL") else 0.0,
                    "notes":            "Project invoice. Payment due NET30.",
                    "source_system":    "QuickBooks Enterprise 2025",
                    "source_system_type": "Billing",
                })

    return invoices

# ── 3. PSA Tickets ────────────────────────────────────────────────────────────

TICKET_TYPES    = ["Incident", "Service Request", "Problem", "Change Request"]
TICKET_PRIORITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TECHNICIANS     = ["Alex Rivera", "Priya Nair", "Tom Okafor", "Mia Chen",
                   "Derek Walsh", "Fatima Hassan", "Luis Morales", "Sam Park"]

TICKET_TEMPLATES = {
    "Incident": [
        "Server unresponsive -- rebooted and monitoring applied",
        "Microsoft 365 email delivery failure -- MX records corrected",
        "VPN connectivity issues -- certificate renewal required",
        "Ransomware attempt detected and quarantined by EDR",
        "Backup job failed -- storage capacity reached, expanded",
        "Network switch port flapping -- replaced SFP module",
        "Workstation BSOD loop -- driver conflict resolved",
        "Multi-factor authentication lockout -- reset and documented",
    ],
    "Service Request": [
        "New employee onboarding -- M365 licence provisioned",
        "Password reset and security policy briefing",
        "Laptop imaging and domain join for new hire",
        "Printer driver deployment via Group Policy",
        "VPN client install and configuration",
        "SharePoint permissions adjustment per manager request",
        "Software licence upgrade -- Adobe Creative Cloud",
        "Remote desktop access configuration",
    ],
    "Problem": [
        "Recurring email sync delays -- root cause: throttling policy",
        "Intermittent Wi-Fi drops in conference rooms -- investigated AP placement",
        "Slow application load times -- SQL query optimisation required",
        "Firewall rule causing intermittent DNS failures",
    ],
    "Change Request": [
        "Firewall rule update for new SaaS application",
        "Server patching -- monthly maintenance window",
        "Active Directory OU restructure per org chart update",
        "Switch VLAN reconfiguration for IoT device isolation",
        "SSL certificate renewal for internal portal",
        "Azure subscription tier upgrade",
    ],
}

def generate_psa_tickets(clients: list, invoices: list) -> list:
    tickets    = []
    tkt_seq    = 5000
    client_map = {c["id"]: c for c in clients}

    # Generate 5-7 tickets per client for the last 3 billing months
    for client in clients:
        for month_offset in range(3):       # Oct, Nov, Dec 2025
            month_start = date(2025, 10 + month_offset, 1) if month_offset < 2 else date(2025, 12, 1)
            month_end   = date(2025, 10 + month_offset, 28) if month_offset < 2 else date(2025, 12, 28)

            num_tickets = random.randint(3, 7)
            for _ in range(num_tickets):
                tkt_seq += 1
                t_type    = random.choice(TICKET_TYPES)
                priority  = random.choices(
                    TICKET_PRIORITIES, weights=[20, 50, 25, 5]
                )[0]
                created   = month_start + timedelta(days=random.randint(0, 27))
                hours     = round(random.uniform(0.5, 6.0), 1)
                resolved  = created + timedelta(
                    hours=random.choice([1, 2, 4, 8, 24, 48])
                )

                # Link to an invoice if one exists for this month
                month_invoices = [
                    inv for inv in invoices
                    if inv["client_id"] == client["id"]
                    and inv["billing_period"] == month_start.strftime("%B %Y")
                ]
                linked_inv = month_invoices[0]["invoice_id"] if month_invoices else None

                title = random.choice(TICKET_TEMPLATES.get(t_type, ["General support request"]))

                tickets.append({
                    "ticket_id":       f"TKT-2025-{tkt_seq:04d}",
                    "client_id":       client["id"],
                    "client_name":     client["name"],
                    "linked_invoice":  linked_inv,
                    "type":            t_type,
                    "priority":        priority,
                    "title":           title,
                    "created_date":    created.isoformat(),
                    "resolved_date":   resolved.isoformat(),
                    "technician":      random.choice(TECHNICIANS),
                    "hours_billed":    hours,
                    "billing_status":  "BILLED" if linked_inv else "UNBILLED",
                    "status":          "CLOSED",
                    "sla_met":         priority != "CRITICAL" or random.random() > 0.1,
                    "resolution_note": f"Issue resolved: {title.lower()}. Client notified.",
                    "source_system":   "ConnectWise Manage",
                    "source_system_type": "PSA",
                })

    return tickets

# ── 4. CRM Profiles ───────────────────────────────────────────────────────────

HEALTH_NOTES = {
    "EXCELLENT": "Account in excellent standing. On-time payments, proactive engagement.",
    "GOOD":      "Healthy account. Occasional late payments but responsive to reminders.",
    "FAIR":      "Account requires monitoring. Payment delays recurring. Escalation risk.",
    "POOR":      "High-risk account. Multiple overdue invoices. Legal review may be required.",
}

def generate_crm_profiles(clients: list, invoices: list) -> list:
    profiles = []
    rel_to_health = {"EXCELLENT": "GREEN", "GOOD": "GREEN",
                     "FAIR": "YELLOW", "POOR": "RED"}

    for client in clients:
        client_invs  = [i for i in invoices if i["client_id"] == client["id"]]
        total_billed = sum(i["total_amount"] for i in client_invs)
        total_paid   = sum(i["amount_paid"]  for i in client_invs)
        open_balance = round(total_billed - total_paid, 2)
        overdue_invs = [i for i in client_invs
                        if i["status"] in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE")]

        health = rel_to_health[client["reliability"]]
        start  = date.fromisoformat(client["start"])
        years  = round((TODAY - start).days / 365.25, 1)

        upsell_options = []
        if "SEC_ADV" not in client["services"] and client["size"] in ("M", "L"):
            upsell_options.append("SOC/SIEM Security Upgrade")
        if "CLOUD" not in client["services"]:
            upsell_options.append("Cloud Infrastructure Migration")
        if "BCP" not in client["services"] and client["industry"] in ("Healthcare", "Finance", "Legal"):
            upsell_options.append("Business Continuity Planning")
        if "COMPLY" not in client["services"] and client["industry"] in ("Healthcare", "Finance"):
            upsell_options.append("Compliance Consulting Package")

        profiles.append({
            "client_id":           client["id"],
            "client_name":         client["name"],
            "industry":            client["industry"],
            "employee_count":      client["users"],
            "client_since":        client["start"],
            "years_as_client":     years,
            "account_manager":     client["manager"],
            "account_health":      health,
            "health_note":         HEALTH_NOTES[client["reliability"]],
            "payment_reliability": client["reliability"],
            "total_billed_ytd":    round(total_billed, 2),
            "total_paid_ytd":      round(total_paid, 2),
            "open_balance":        open_balance,
            "overdue_invoice_count": len(overdue_invs),
            "overdue_balance":     round(sum(i["balance_due"] for i in overdue_invs), 2),
            "upsell_opportunities": upsell_options,
            "last_qbr_date":       (TODAY - timedelta(days=random.randint(30, 180))).isoformat(),
            "nps_score":           random.choices([9, 10, 8, 7, 6, 5, 4],
                                                  weights=[20, 15, 25, 15, 10, 10, 5])[0],
            "contacts":            client["contacts"],
            "address":             client["address"],
            "source_system":       "HubSpot CRM",
            "source_system_type":  "CRM",
        })

    return profiles

# ── 5. Communications Log ─────────────────────────────────────────────────────

REMINDER_SUBJECTS = {
    1: "Friendly Reminder: Invoice {inv_id} Due {days} Days Ago",
    2: "Second Notice: Invoice {inv_id} Now {days} Days Past Due",
    3: "Final Notice: Invoice {inv_id} -- Immediate Payment Required",
    4: "URGENT: Account {client} Referred to Collections Review",
}

REMINDER_BODIES = {
    1: ("Dear {contact},\n\nThis is a friendly reminder that invoice {inv_id} "
        "for ${amount:,.2f} was due on {due_date}. If you have already sent payment, "
        "please disregard this notice. Otherwise, please remit payment at your earliest convenience.\n\n"
        "Thank you for your continued business.\n\nTechVault MSP Billing Team"),
    2: ("Dear {contact},\n\nOur records indicate invoice {inv_id} for ${amount:,.2f} "
        "is now {days} days past due. Please arrange payment immediately to avoid service interruption "
        "and the accrual of late fees (1.5% per month).\n\n"
        "If there is a dispute or billing question, please contact us immediately.\n\n"
        "TechVault MSP Billing Team"),
    3: ("Dear {contact},\n\nDespite previous notices, invoice {inv_id} for ${amount:,.2f} "
        "remains unpaid at {days} days past due. This is your final notice before this account "
        "is referred for collections review.\n\n"
        "Payment in full is required within 5 business days to avoid service suspension "
        "and legal action.\n\nTechVault MSP -- Accounts Receivable"),
    4: ("This account has been flagged for collections review. "
        "Invoice {inv_id} outstanding {days} days. Balance: ${amount:,.2f}."),
}

RESPONSE_TYPES = [
    "NO_RESPONSE",
    "ACKNOWLEDGED_WILL_PAY",
    "PAYMENT_PLAN_REQUESTED",
    "DISPUTE_RAISED",
    "PARTIAL_PAYMENT_PROMISED",
]

def generate_communications(clients: list, invoices: list) -> list:
    comms     = []
    comm_seq  = 3000
    client_map = {c["id"]: c for c in clients}

    for inv in invoices:
        if inv["status"] not in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE"):
            continue

        days_out = inv["days_outstanding"]
        client   = client_map.get(inv["client_id"], {})
        contact  = client.get("contacts", {}).get("ar_contact", {}).get("name", "Billing Contact")
        email    = inv["billing_contact"]
        due_dt   = date.fromisoformat(inv["due_date"])

        # Determine how many reminders have been sent based on days outstanding
        num_reminders = 0
        if days_out >= 10:  num_reminders = 1
        if days_out >= 30:  num_reminders = 2
        if days_out >= 60:  num_reminders = 3
        if days_out >= 90:  num_reminders = 4

        for seq in range(1, num_reminders + 1):
            comm_seq += 1
            # Reminder sent N days after due date
            reminder_offsets = {1: 10, 2: 30, 3: 60, 4: 90}
            sent_date  = due_dt + timedelta(days=reminder_offsets[seq])
            subject    = REMINDER_SUBJECTS[seq].format(
                inv_id=inv["invoice_id"], days=days_out,
                client=inv["client_name"],
            )
            body = REMINDER_BODIES[seq].format(
                contact=contact, inv_id=inv["invoice_id"],
                amount=inv["balance_due"], due_date=inv["due_date"], days=days_out,
                client=inv["client_name"],
            )
            # Response probability declines with each notice
            response_p = [0.0, 0.35, 0.25, 0.15, 0.10][seq]
            response   = random.choice(RESPONSE_TYPES[1:]) if random.random() < response_p \
                         else "NO_RESPONSE"

            comms.append({
                "comm_id":         f"COMM-2025-{comm_seq:04d}",
                "invoice_id":      inv["invoice_id"],
                "client_id":       inv["client_id"],
                "client_name":     inv["client_name"],
                "sent_date":       sent_date.isoformat(),
                "reminder_sequence": seq,
                "channel":         "email",
                "direction":       "OUTBOUND",
                "sent_from":       MSP["billing_email"],
                "sent_to":         email,
                "subject":         subject,
                "body_preview":    body[:400],
                "status":          "DELIVERED",
                "client_response": response,
                "invoice_amount":  inv["total_amount"],
                "balance_at_time": inv["balance_due"],
                "days_overdue_at_send": days_out,
                "account_manager": inv["account_manager"],
                "source_system":   "Microsoft Exchange Online",
                "source_system_type": "Communications",
            })

    return comms

# ── 6. Contracts ──────────────────────────────────────────────────────────────

CONTRACT_TYPES = {
    "S": "Managed Services Agreement - Standard",
    "M": "Managed Services Agreement - Professional",
    "L": "Managed Services Agreement - Enterprise",
}

def generate_contracts(clients: list) -> list:
    contracts = []
    for i, client in enumerate(clients, 1):
        start   = date.fromisoformat(client["start"])
        term_yr = random.choice([1, 2, 3])
        end     = start.replace(year=start.year + term_yr)
        if end < TODAY:
            # Auto-renewed
            while end < TODAY:
                end = end.replace(year=end.year + term_yr)

        monthly_val = sum(
            SERVICES[s]["price"] * (client["users"] if SERVICES[s]["type"] == "per_user" else 1)
            for s in client["services"]
            if SERVICES[s]["type"] not in ("project", "hourly")
        )
        monthly_val = round(monthly_val, 2)

        contracts.append({
            "contract_id":          f"MSA-{str(start.year)[-2:]}-{i:03d}",
            "client_id":            client["id"],
            "client_name":          client["name"],
            "contract_type":        CONTRACT_TYPES[client["size"]],
            "effective_date":       start.isoformat(),
            "expiry_date":          end.isoformat(),
            "auto_renew":           True,
            "renewal_notice_days":  90,
            "payment_terms":        client["terms"],
            "monthly_value":        monthly_val,
            "annual_value":         round(monthly_val * 12, 2),
            "services_in_scope":    client["services"],
            "sla_response_time":    {"CRITICAL": "1 hour", "HIGH": "4 hours",
                                     "MEDIUM": "8 hours", "LOW": "next business day"},
            "sla_uptime_guarantee": "99.5%",
            "late_payment_penalty": "1.5% per month on outstanding balance after due date",
            "suspension_clause":    "Services may be suspended after 60 days of non-payment "
                                    "with 5 business days written notice.",
            "termination_notice":   "90 days written notice required by either party.",
            "governing_law":        "State of California",
            "signed_by_client":     client["contacts"]["cfo"]["name"],
            "signed_by_msp":        "Robert Vance, CEO -- TechVault MSP",
            "source_system":        "SharePoint Contract Repository",
            "source_system_type":   "Contracts",
        })

    return contracts

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = Path("data/enterprise")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Enriching client profiles...")
    clients = enrich_clients(CLIENT_TEMPLATES)

    print("Generating invoices...")
    invoices = generate_invoices(clients)

    print("Generating PSA tickets...")
    tickets = generate_psa_tickets(clients, invoices)

    print("Generating CRM profiles...")
    crm = generate_crm_profiles(clients, invoices)

    print("Generating communications log...")
    comms = generate_communications(clients, invoices)

    print("Generating contracts...")
    contracts = generate_contracts(clients)

    # ── Wrap in source-system envelopes ──────────────────────────────────────
    datasets = {
        "clients.json":        {"source_system": "TechVault MSP Master Client List", "records": clients},
        "invoices.json":       {"source_system": "QuickBooks Enterprise 2025",        "records": invoices},
        "psa_tickets.json":    {"source_system": "ConnectWise Manage",                "records": tickets},
        "crm_profiles.json":   {"source_system": "HubSpot CRM",                      "records": crm},
        "communications.json": {"source_system": "Microsoft Exchange Online",         "records": comms},
        "contracts.json":      {"source_system": "SharePoint Contract Repository",    "records": contracts},
    }

    for fname, payload in datasets.items():
        path = out_dir / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"  Wrote {len(payload['records']):>4} records -> {path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    overdue = [i for i in invoices if i["status"] in ("OVERDUE", "OVERDUE_CRITICAL", "PAST_DUE")]
    critical = [i for i in invoices if i["status"] == "OVERDUE_CRITICAL"]
    total_ar  = sum(i["balance_due"] for i in invoices if i["status"] != "PAID")

    print()
    print("=" * 55)
    print("  Enterprise Data Generation Complete")
    print("=" * 55)
    print(f"  Clients          : {len(clients)}")
    print(f"  Invoices (total) : {len(invoices)}")
    print(f"    PAID           : {sum(1 for i in invoices if i['status'] == 'PAID')}")
    print(f"    OPEN           : {sum(1 for i in invoices if i['status'] == 'OPEN')}")
    print(f"    Past Due 1-30d : {sum(1 for i in invoices if i['status'] == 'PAST_DUE')}")
    print(f"    OVERDUE 31-90d : {sum(1 for i in invoices if i['status'] == 'OVERDUE')}")
    print(f"    CRITICAL 90d+  : {len(critical)}")
    print(f"    Total AR Open  : ${total_ar:,.2f}")
    print(f"  PSA Tickets      : {len(tickets)}")
    print(f"  CRM Profiles     : {len(crm)}")
    print(f"  Communications   : {len(comms)}")
    print(f"  Contracts        : {len(contracts)}")
    print(f"  Invoices >60 days: {len([i for i in overdue if i['days_outstanding'] >= 60])}")
    print("=" * 55)


if __name__ == "__main__":
    main()
