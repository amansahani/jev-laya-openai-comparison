"""
Challenging Indian Finance Benchmark Dataset.
Covers:
1. Indian Financial Regulators & Jurisdictions (RBI, SEBI, IRDAI, FIU-IND / PMLA, PFRDA).
2. RBI Asset Quality & NPA/SMA Classification (Prudential Framework on Resolution of Stressed Assets).
3. Cyber Financial Fraud & Modus Operandi Detection (Digital Arrest, Loan App Extortion, UPI Phishing, Mule Accounts).
4. Indian Tax & FEMA Compliance (TDS/TCS mismatch, Foreign Inward Remittance LRS limits, STCG/LTCG).
"""

from typing import List, Dict, Any

INDIAN_FINANCE_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "IFB-01",
        "category": "Regulatory Jurisdiction",
        "scenario": "A listed NBFC's promoter shared unpublished price-sensitive financial results (UPSI) with an informal Telegram group 2 hours before the board meeting disclosure to BSE/NSE.",
        "ground_truth": {
            "regulator": "sebi",
            "is_severe_violation": True,
            "severity_score": 3  # Critical
        },
        "questions": {
            "regulator": {
                "type": "choice",
                "instructions": "Identify the statutory regulatory authority with primary jurisdiction to investigate this violation.",
                "criteria": {
                    "sebi": "Securities and Exchange Board of India (insider trading, stock manipulation, listed disclosures, PIT regulations)",
                    "rbi": "Reserve Bank of India (banking operations, monetary policy, lending guidelines, payment systems)",
                    "irdai": "Insurance Regulatory and Development Authority of India (insurance policies, claims, solvency)",
                    "fiu_ind": "Financial Intelligence Unit India (anti-money laundering, cash reporting, STR filings)"
                }
            },
            "is_severe_violation": {
                "type": "noul",
                "instructions": "Does this represent a major statutory violation requiring immediate regulatory enforcement action?"
            },
            "severity_score": {
                "type": "score",
                "instructions": "Rate regulatory severity from 0 (minor technical defect) to 3 (critical market misconduct / criminal liability).",
                "criteria": ["0: minor procedural lapse", "1: moderate administrative delay", "2: major compliance failure", "3: critical statutory violation / criminal offence"]
            }
        }
    },
    {
        "id": "IFB-02",
        "category": "Asset Classification (RBI)",
        "scenario": "An MSME borrower with a working capital CC limit of ₹5 Crores has continuous interest overdue for 48 days. The account has not been serviced since the last quarter.",
        "ground_truth": {
            "asset_status": "sma_1",
            "is_npa": False,
            "risk_level": 2  # High Risk (SMA-1)
        },
        "questions": {
            "asset_status": {
                "type": "choice",
                "instructions": "Classify this loan account under the RBI Prudential Framework for Asset Classification and Resolution of Stressed Assets based on 48 days past due.",
                "criteria": {
                    "standard": "Standard Asset (0 days overdue)",
                    "sma_0": "Special Mention Account-0 (Principal or interest payment overdue between 1 to 30 days)",
                    "sma_1": "Special Mention Account-1 (Principal or interest payment overdue between 31 to 60 days)",
                    "sma_2": "Special Mention Account-2 (Principal or interest payment overdue between 61 to 90 days)",
                    "npa": "Non-Performing Asset (>90 days overdue / out of order)"
                }
            },
            "is_npa": {
                "type": "noul",
                "instructions": "Has this account officially crossed the threshold into a Non-Performing Asset (NPA) under RBI guidelines?"
            },
            "risk_level": {
                "type": "score",
                "instructions": "Rate the credit distress level from 0 (healthy) to 3 (default/loss).",
                "criteria": ["0: standard healthy asset", "1: early stress (SMA-0)", "2: significant delinquency (SMA-1/SMA-2)", "3: defaulted non-performing asset (NPA)"]
            }
        }
    },
    {
        "id": "IFB-03",
        "category": "Cyber Financial Crime",
        "scenario": "A victim received a WhatsApp video call from individuals dressed in police uniforms claiming their Aadhaar was linked to a money laundering case in Mumbai. They coerced the victim to transfer ₹18 Lakhs to a 'Supreme Court Verification Escrow Account' via RTGS.",
        "ground_truth": {
            "fraud_type": "digital_arrest",
            "is_emergency_fraud": True,
            "urgency": 3
        },
        "questions": {
            "fraud_type": {
                "type": "choice",
                "instructions": "Classify the specific cyber fraud modus operandi according to Indian Cyber Crime Coordination Centre (I4C) taxonomy.",
                "criteria": {
                    "digital_arrest": "Digital arrest scam (impersonation of police, CBI, ED, court officials via video call demanding fund transfers)",
                    "part_time_job": "Part-time task scam (Telegram ratings, YouTube like/subscribe task fraud)",
                    "loan_app_extortion": "Instant Chinese/illegal loan app harassment with morphed photos",
                    "upi_qr_phishing": "UPI collect request / false payment QR code scanning scam"
                }
            },
            "is_emergency_fraud": {
                "type": "noul",
                "instructions": "Does this require urgent reporting to the 1930 Cyber Fraud Helpline and bank nodal officers for immediate freeze of beneficiary mule accounts?"
            },
            "urgency": {
                "type": "score",
                "instructions": "Rate the financial escalation urgency.",
                "criteria": ["0: informational", "1: low priority", "2: urgent dispute", "3: emergency cybercrime active fund freeze required"]
            }
        }
    },
    {
        "id": "IFB-04",
        "category": "Cross-Border & FEMA",
        "scenario": "A resident Indian individual remitted $350,000 USD overseas in a single financial year to buy unlisted shares of a private entity in Singapore without obtaining prior RBI approval.",
        "ground_truth": {
            "fema_violation": "lrs_limit_exceeded",
            "is_fema_breach": True,
            "penalty_risk": 3
        },
        "questions": {
            "fema_violation": {
                "type": "choice",
                "instructions": "Determine the regulatory breach under RBI's Liberalised Remittance Scheme (LRS) and Foreign Exchange Management Act (FEMA).",
                "criteria": {
                    "lrs_limit_exceeded": "LRS annual ceiling breach (exceeded the statutory $250,000 USD per financial year limit without RBI approval)",
                    "permissible_remittance": "Permissible capital account transaction within statutory limits",
                    "tcs_exemption": "Tax collected at source procedural defect only",
                    "fcra_violation": "Foreign Contribution Regulation Act charitable trust violation"
                }
            },
            "is_fema_breach": {
                "type": "noul",
                "instructions": "Does this remittance constitute an unlawful breach of the FEMA Liberalised Remittance Scheme threshold?"
            },
            "penalty_risk": {
                "type": "score",
                "instructions": "Rate the enforcement and compounding penalty risk under FEMA.",
                "criteria": ["0: zero penalty", "1: minor late submission fee", "2: moderate compliance penalty", "3: severe compounding penalty / ED adjudication"]
            }
        }
    },
    {
        "id": "IFB-05",
        "category": "Insurance Mis-selling & Claims (IRDAI)",
        "scenario": "A private bank branch manager told an 82-year-old pensioner that a ₹5 Lakh fixed deposit comes with free health insurance, but secretly locked the funds into a 10-year regular premium Unit Linked Insurance Plan (ULIP).",
        "ground_truth": {
            "violation_type": "bancassurance_mis_selling",
            "is_unfair_practice": True,
            "grievance_severity": 3
        },
        "questions": {
            "violation_type": {
                "type": "choice",
                "instructions": "Classify the grievance under IRDAI Protection of Policyholders' Interests Regulations.",
                "criteria": {
                    "bancassurance_mis_selling": "Mis-selling / fraudulent conversion of bank deposits into insurance products by bancassurance agents",
                    "claim_repudiation": "Legitimate claim rejection based on pre-existing disease non-disclosure",
                    "lapsed_policy": "Routine policy lapsation due to non-payment of renewal premium",
                    "underwriting_rejection": "Medical underwriting refusal"
                }
            },
            "is_unfair_practice": {
                "type": "noul",
                "instructions": "Does this constitute unfair trade practice and deceptive mis-selling under IRDAI norms?"
            },
            "grievance_severity": {
                "type": "score",
                "instructions": "Rate the consumer protection severity level.",
                "criteria": ["0: routine query", "1: minor service delay", "2: administrative dispute", "3: severe predatory mis-selling to vulnerable senior citizen"]
            }
        }
    },
    {
        "id": "IFB-06",
        "category": "UPI & Payment Systems (RBI Ombudsman)",
        "scenario": "A customer attempted a ₹25,000 UPI transaction at a jeweler. The money was debited from their SBI account, the transaction failed at the payment gateway, and the funds were not auto-reversed after 8 business days.",
        "ground_truth": {
            "ombudsman_remedy": "tat_compensation_due",
            "is_eligible_for_compensation": True,
            "delay_severity": 2
        },
        "questions": {
            "ombudsman_remedy": {
                "type": "choice",
                "instructions": "What is the primary regulatory entitlement under the RBI Harmonisation of Turn Around Time (TAT) and customer compensation framework for failed transactions?",
                "criteria": {
                    "tat_compensation_due": "Mandatory compensation of ₹100 per day of delay beyond T+1 turnaround time",
                    "chargeback_denial": "Permanent forfeiture of transaction amount",
                    "merchant_liability_only": "Merchant alone is liable, bank has zero responsibility",
                    "court_case_only": "Only remedy is civil court litigation"
                }
            },
            "is_eligible_for_compensation": {
                "type": "noul",
                "instructions": "Is the customer statutorily eligible for auto-compensation under RBI's T+1 TAT framework?"
            },
            "delay_severity": {
                "type": "score",
                "instructions": "Rate the service deficiency under RBI Banking Ombudsman criteria.",
                "criteria": ["0: within permissible TAT (T+1)", "1: minor delay (T+2/T+3)", "2: major TAT breach (>5 days delay)", "3: prolonged unresolved dispute (>30 days)"]
            }
        }
    },
    {
        "id": "IFB-07",
        "category": "Tax Deducted at Source (Income Tax Act)",
        "scenario": "A buyer purchased residential property in Bangalore for ₹85 Lakhs from a resident seller but failed to deduct 1% TDS under Section 194-IA before remitting the full consideration.",
        "ground_truth": {
            "tax_section": "section_194ia_default",
            "is_tax_default": True,
            "compliance_risk": 2
        },
        "questions": {
            "tax_section": {
                "type": "choice",
                "instructions": "Identify the non-compliance section under the Indian Income Tax Act, 1961.",
                "criteria": {
                    "section_194ia_default": "Default in deducting 1% TDS on immovable property transfer exceeding ₹50 Lakhs (Section 194-IA)",
                    "section_194c": "Contractor TDS default under Section 194C",
                    "section_194j": "Professional technical fees TDS default under Section 194J",
                    "section_192": "Salary TDS default under Section 192"
                }
            },
            "is_tax_default": {
                "type": "noul",
                "instructions": "Does the buyer become an 'assessee-in-default' liable for interest and penalty under Section 201?"
            },
            "compliance_risk": {
                "type": "score",
                "instructions": "Rate the tax audit risk.",
                "criteria": ["0: fully compliant", "1: procedural delay with TDS deducted", "2: failure to deduct TDS on high-value transaction", "3: intentional tax evasion scheme"]
            }
        }
    },
    {
        "id": "IFB-08",
        "category": "Stock Market / Finfluencer Regulation (SEBI)",
        "scenario": "An unregistered social media finfluencer with 1.2M followers accepted undisclosed ₹10 Lakh sponsorship from a micro-cap company to post bullish buy targets, generating artificial volume before dumping their personal holdings.",
        "ground_truth": {
            "market_abuse": "pump_and_dump_fraud",
            "is_sebi_violation": True,
            "enforcement_level": 3
        },
        "questions": {
            "market_abuse": {
                "type": "choice",
                "instructions": "Categorize this conduct under SEBI (Prohibition of Fraudulent and Unfair Trade Practices) Regulations (PFUTP).",
                "criteria": {
                    "pump_and_dump_fraud": "Pump and dump market manipulation / front-running by unregistered investment adviser with undisclosed conflict",
                    "genuine_educational_content": "Bona fide educational commentary without stock recommendations",
                    "research_analyst_report": "SEBI-registered Research Analyst compliant disclosure",
                    "algorithmic_glitch": "Technical connectivity latency glitch"
                }
            },
            "is_sebi_violation": {
                "type": "noul",
                "instructions": "Is this a direct violation of SEBI PFUTP regulations and Research Analyst Regulations?"
            },
            "enforcement_level": {
                "type": "score",
                "instructions": "Rate the probability of SEBI ex-parte interim order and disgorgement of unlawful gains.",
                "criteria": ["0: no action", "1: advisory warning letter", "2: administrative settlement with fee", "3: market ban, impounding of profits, and criminal prosecution"]
            }
        }
    },
    {
        "id": "IFB-09",
        "category": "Anti-Money Laundering & Mule Accounts (FIU-IND)",
        "scenario": "A newly opened savings account of a college student in Tier-3 town with declared annual income of ₹50,000 received 42 credits of ₹49,500 each via IMPS in 48 hours, which were instantly withdrawn in cash at ATMs.",
        "ground_truth": {
            "aml_pattern": "smurfing_mule_account",
            "is_str_mandated": True,
            "suspicion_score": 3
        },
        "questions": {
            "aml_pattern": {
                "type": "choice",
                "instructions": "Classify the suspicious transaction pattern under PMLA & RBI KYC/AML Master Directions.",
                "criteria": {
                    "smurfing_mule_account": "Structuring / smurfing transactions just below ₹50,000 PAN threshold via money mule account",
                    "regular_retail_activity": "Normal student scholarship and family maintenance remittances",
                    "corporate_payroll": "Legitimate institutional salary batch disbursement",
                    "inter_bank_clearing": "Routine clearing house adjustment"
                }
            },
            "is_str_mandated": {
                "type": "noul",
                "instructions": "Is the bank's Principal Officer legally mandated to file a Suspicious Transaction Report (STR) to FIU-IND?"
            },
            "suspicion_score": {
                "type": "score",
                "instructions": "Rate the AML risk score.",
                "criteria": ["0: low risk", "1: medium risk / explainable", "2: high risk requiring enhanced due diligence", "3: extreme suspicion / confirmed cyber fraud mule"]
            }
        }
    },
    {
        "id": "IFB-10",
        "category": "Hinglish Customer Dispute (FinTech / NBFC)",
        "scenario": "Maine loan app se 5,000 rupaye liye the jo maine repay kar diye. Ab recovery agents mere pure contacts ko phone karke meri morphed photo bhej rahe hain aur 25,000 mang rahe hain.",
        "ground_truth": {
            "dispute_nature": "illegal_lending_extortion",
            "is_harassment": True,
            "police_escalation": 3
        },
        "questions": {
            "dispute_nature": {
                "type": "choice",
                "instructions": "Analyze this grievance reported in colloquial Hinglish and determine its legal/operational category.",
                "criteria": {
                    "illegal_lending_extortion": "Illegal predatory digital lending app extortion and criminal intimidation in violation of RBI Digital Lending Guidelines",
                    "routine_emi_reminder": "Standard courteous EMI reminder call by regulated bank",
                    "kyc_update_request": "Routine periodic KYC re-verification message",
                    "credit_score_inquiry": "General consumer credit report score inquiry"
                }
            },
            "is_harassment": {
                "type": "noul",
                "instructions": "Does this violate RBI's Fair Practices Code and Digital Lending guidelines on recovery agent conduct?"
            },
            "police_escalation": {
                "type": "score",
                "instructions": "Rate the urgency of police/cyber cell FIR escalation.",
                "criteria": ["0: no grievance", "1: minor customer service complaint", "2: internal ombudsman escalation", "3: immediate criminal FIR and cyber cell complaint needed"]
            }
        }
    }
]
