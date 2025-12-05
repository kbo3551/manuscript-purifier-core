#!/usr/bin/env python
# coding: utf-8

# PubMed XML 파서 + SpaCy NER 기반 기관/국가 정제 + DB Insert 전체 파이프라인 (GPU + 배치 처리 최적화)

import os
import re
import spacy
import pandas as pd
from tqdm import tqdm
from lxml import etree as ET
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
import time
import torch
import gc

# 파일 경로
ORG_EXCEL_PATH = "/home/mediazen/pubmed_temp_py/files/20250704_084016/organization_리스트/organization_ror167_250703.xlsx"
SUB_ORG_EXCEL_PATH = "/home/mediazen/pubmed_temp_py/files/20250704_084016/organization_리스트/sub_organization_250703.xlsx"

# 전역
engine = None
metadata = None
parsed_pmids = set()

# SpaCy + EntityRuler + 조직명 정제 사전 준비
def setup_spacy_and_dict_with_ror(org_path, sub_org_path):
    nlp = spacy.load("en_core_web_trf", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    df_org = pd.read_excel(org_path)
    df_sub = pd.read_excel(sub_org_path)
    df_org = df_org[df_org['list'] == 'Keep']

    df_org['clean_org'] = df_org['raw_org'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', str(x)).upper())
    df_sub['clean_sub_org'] = df_sub['raw_sub_org'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', str(x)).upper())

    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True}, after="ner")
    patterns = ([{"label": "ORG", "pattern": name} for name in df_org['raw_org'].drop_duplicates()] +
                [{"label": "ORG", "pattern": name} for name in df_org['clean_org'].drop_duplicates()])
    ruler.add_patterns(patterns)

    clean_org = set(df_org.clean_org.drop_duplicates().to_list())
    clean_sub_org = set(df_sub.clean_sub_org.drop_duplicates().to_list())
    raw_org = set(df_org.raw_org.drop_duplicates().str.upper().to_list())
    raw_sub_org = set(df_sub.raw_sub_org.drop_duplicates().str.upper().to_list())

    df_org['clean_std_org'] = df_org['std_org'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', str(x)).upper())
    org_country_map = df_org.dropna(subset=['clean_org', 'clean_std_org', 'country_name']) \
                            .drop_duplicates(subset=['clean_org']) \
                            .set_index('clean_org')[['clean_std_org', 'country_name']] \
                            .to_dict(orient='index')

    return nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map

# SpaCy 초기화 함수
def init_spacy():
    global GLOBAL_NLP_DATA
    if torch.cuda.is_available():
        spacy.require_gpu()
        print(" GPU 사용: ", spacy.prefer_gpu())
    else:
        spacy.require_cpu()
        print(" GPU 사용 불가 → CPU 처리")
    GLOBAL_NLP_DATA = setup_spacy_and_dict_with_ror(ORG_EXCEL_PATH, SUB_ORG_EXCEL_PATH)

# 배치 주소 정제 함수
def batch_refine_affiliations(address_list, nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map):
    #docs = nlp.pipe(address_list, batch_size=32) 메모리 누수로 인한 list 감싸기
    docs = list(nlp.pipe(address_list, batch_size=64))
    results = []
    for address, doc in zip(address_list, docs):
        ner_ORG = {ent.text for ent in doc.ents if ent.label_ == "ORG"}
        list_organizations, list_sub_organizations = set(), set()
        for j in ner_ORG:
            clean_j = re.sub(r'[^A-Za-z0-9\s]', '', j).upper()
            if clean_j in clean_org or j.upper() in raw_org:
                list_organizations.add(j)
            elif clean_j in clean_sub_org or j.upper() in raw_sub_org:
                list_sub_organizations.add(j)
        first_organization, country_name = None, None
        if list_organizations:
            org_pos = [(org, address.find(org)) for org in list_organizations if address.find(org) != -1]
            if org_pos:
                first_organization = sorted(org_pos, key=lambda x: x[1])[0][0]
                clean_first = re.sub(r'[^A-Za-z0-9\s]', '', first_organization).upper()
                if clean_first in org_country_map:
                    country_name = org_country_map[clean_first]["country_name"]
        results.append((address, first_organization, country_name))
    return results

def parse_pubmed_chunk(xml_path, start_offset=0, chunk_limit=100):
    global GLOBAL_NLP_DATA
    nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map = GLOBAL_NLP_DATA
    results = []
    context = ET.iterparse(xml_path, events=('end',), tag='PubmedArticle', recover=True, encoding='utf-8')
    count = 0

    for i, (_, elem) in enumerate(context):
        try:
            if i < start_offset:
                continue

            pmid = elem.findtext(".//PMID")
            if not pmid or pmid in parsed_pmids:
                continue

            parsed_pmids.add(pmid)

            # 소속 먼저 추출
            affiliations = [a.text for a in elem.findall(".//Affiliation") if a.text]
            if not affiliations:
                results.append({"pmid": pmid, "skip_reason": "missing_affiliation"})
                continue

            refined_affs = batch_refine_affiliations(
                affiliations, nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map
            )

            if not any(org for _, org, _ in refined_affs):
                results.append({"pmid": pmid, "skip_reason": "empty_org_country"})
                continue

            # 날짜 처리
            pubdate = elem.find(".//PubDate")
            year = pubdate.findtext("Year") if pubdate is not None else None
            month = pubdate.findtext("Month") if pubdate is not None else None
            day = pubdate.findtext("Day") if pubdate is not None else None

            # 제목
            article_title = elem.findtext(".//ArticleTitle")

            # 키워드
            keywords = [k.text for k in elem.findall(".//KeywordList/Keyword") if k.text]

            # Journal 정보
            journal_elem = elem.find(".//Journal")
            journal_title = journal_elem.findtext("Title") if journal_elem is not None else None
            journal_iso_abbreviation = journal_elem.findtext("ISOAbbreviation") if journal_elem is not None else None
            journal_volume = elem.findtext(".//JournalIssue/Volume")
            journal_issue = elem.findtext(".//JournalIssue/Issue")
            journal_pubdate = elem.find(".//JournalIssue/PubDate")
            journal_pub_year = journal_pubdate.findtext("Year") if journal_pubdate is not None else None
            journal_pub_month = journal_pubdate.findtext("Month") if journal_pubdate is not None else None
            journal_pub_day = journal_pubdate.findtext("Day") if journal_pubdate is not None else None

            # DataBank 정보
            databanks = []
            databank_list_elem = elem.find(".//DataBankList")
            databank_list_complete_yn = databank_list_elem.attrib.get("CompleteYN") if databank_list_elem is not None else None
            for databank in elem.findall(".//DataBankList/DataBank"):
                databank_name = databank.findtext("DataBankName")
                accession_numbers = [a.text for a in databank.findall("AccessionNumberList/AccessionNumber") if a.text]
                if databank_name:
                    for acc in accession_numbers:
                        databanks.append({
                            "databank_name": databank_name,
                            "accession_number": acc,
                            "complete_yn": databank_list_complete_yn
                        })

            # MESH
            mesh_terms = []
            for mesh in elem.findall(".//MeshHeading"):
                desc = mesh.find("DescriptorName")
                desc_name = desc.text if desc is not None else None
                desc_ui = desc.attrib.get("UI") if desc is not None else None
                desc_maj = desc.attrib.get("MajorTopicYN", "N") if desc is not None else "N"
                qualifiers = mesh.findall("QualifierName")
                if qualifiers:
                    for q in qualifiers:
                        mesh_terms.append({
                            "DescriptorName": desc_name,
                            "DescriptorUI": desc_ui,
                            "DescriptorMajorTopicYN": desc_maj,
                            "QualifierName": q.text,
                            "QualifierUI": q.attrib.get("UI"),
                            "QualifierMajorTopicYN": q.attrib.get("MajorTopicYN", "N")
                        })
                else:
                    mesh_terms.append({
                        "DescriptorName": desc_name,
                        "DescriptorUI": desc_ui,
                        "DescriptorMajorTopicYN": desc_maj,
                        "QualifierName": None,
                        "QualifierUI": None,
                        "QualifierMajorTopicYN": None
                    })

            # 저자 처리
            authors = []
            for a in elem.findall(".//AuthorList/Author"):
                lastname = a.findtext("LastName")
                forename = a.findtext("ForeName")

                if not lastname or not forename:
                    print(" 이름 누락 → 스킵")
                    continue

                print(f" Author Candidate: {lastname} {forename}")

                affiliation_texts = [aff.text for aff in a.findall("AffiliationInfo/Affiliation") if aff.text]
                if not affiliation_texts:
                    print(" affiliation 없음 → 스킵:", lastname, forename)
                    continue

                first_aff = affiliation_texts[0]
                refined_aff = batch_refine_affiliations(
                    [first_aff], nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map
                )[0]
                aff_text, org, country = refined_aff

                if not org or not country:
                    print(" org/country 없음 → 스킵:", lastname, forename)
                    continue

                print(f" author 추가: {lastname} {forename} | Org: {org} | Country: {country}")

                authors.append({
                    "LastName": lastname.strip(),
                    "ForeName": forename.strip(),
                    "Initials": a.findtext("Initials"),
                    "Suffix": a.findtext("Suffix"),
                    "CollectiveName": a.findtext("CollectiveName"),
                    "Identifier": a.findtext("Identifier"),
                    "IdentifierSource": a.find("Identifier").attrib.get("Source") if a.find("Identifier") is not None else None,
                    "affiliation_text": first_aff,
                    "organization": org,
                    "country": country
                })

            # 참고문헌
            references = []
            for ref in elem.findall(".//Reference"):
                citation = ref.findtext("Citation")
                ids = {id.attrib.get("IdType"): id.text for id in ref.findall(".//ArticleId")}
                references.append({
                    "citation": citation,
                    "pubmed": ids.get("pubmed"),
                    "doi": ids.get("doi"),
                    "pmcid": ids.get("pmc"),
                    "pii": ids.get("pii")
                })

            results.append({
                "pmid": pmid,
                "affiliations": refined_affs,
                "keywords": keywords,
                "references": references,
                "mesh": mesh_terms,
                "year": year,
                "month": month,
                "day": day,
                "article_title": article_title,
                "authors": authors,
                "journal_title": journal_title,
                "journal_iso_abbreviation": journal_iso_abbreviation,
                "journal_volume": journal_volume,
                "journal_issue": journal_issue,
                "journal_pub_year": journal_pub_year,
                "journal_pub_month": journal_pub_month,
                "journal_pub_day": journal_pub_day,
                "databanks": databanks,
                "skip_reason": None
            })

            count += 1
            if count >= chunk_limit:
                break

        finally:
            elem.clear()
            while elem.getprevious() is not None:
                try:
                    del elem.getparent()[0]
                except Exception:
                    break

    return results



# DB Insert 함수
def get_next_seq_batch(conn, seq_name, batch_size):
    start_val = conn.execute(text(f"SELECT NEXT VALUE FOR {seq_name}")).scalar()
    return list(range(start_val, start_val + batch_size))

def get_next_seq_iter(conn, seq_name, count):
    start_val = conn.execute(text(f"SELECT NEXT VALUE FOR {seq_name}")).scalar()
    for i in range(start_val, start_val + count):
        yield i

def chunked_insert(conn, table, data_list, chunk_size=1000):
    for i in range(0, len(data_list), chunk_size):
        chunk = data_list[i:i+chunk_size]
        conn.execute(table.insert().prefix_with("IGNORE"), chunk)


def insert_all_to_db(parsed_data, engine, db_url):
    global metadata

    BATCH_SIZE = 1000
    total = len(parsed_data)

    count = {
        "pmid": 0, "title": 0, "pubdate": 0, "affiliation": 0, "author": 0,
        "author_aff": 0, "keyword": 0, "reference": 0, "mesh": 0, "journal": 0, "databank": 0
    }

    for batch_start in range(0, total, BATCH_SIZE):
        seen_authors = set()
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_raw = parsed_data[batch_start:batch_end]
        batch = [
            e for e in batch_raw
            if all(org and country for _, org, country in e.get("affiliations", []))
        ]
        print(f"\n🛠️ DB Insert Batch: {batch_start} ~ {batch_end - 1}")

        with engine.begin() as conn:
            memory_guard(max_gb=130)

            # pmid
            pmid_values = [{"pmid": e["pmid"]} for e in batch]
            conn.execute(metadata.tables["pub_pmid_list"].insert().prefix_with("IGNORE"), pmid_values)
            count["pmid"] += len(pmid_values)

            # title
            memory_guard(max_gb=130)
            title_values = [{"pmid": e["pmid"], "article_title": e["article_title"]} for e in batch if e["article_title"]]
            if title_values:
                conn.execute(metadata.tables["pub_article_title"].insert().prefix_with("IGNORE"), title_values)
                count["title"] += len(title_values)

            # pubdate
            memory_guard(max_gb=130)
            pubdate_values = [{"pmid": e["pmid"], "year": e["year"], "month": e["month"], "day": e["day"]} for e in batch if e["year"]]
            if pubdate_values:
                conn.execute(metadata.tables["pub_pubdate"].insert().prefix_with("IGNORE"), pubdate_values)
                count["pubdate"] += len(pubdate_values)

            total_affiliations = sum(len(e["affiliations"]) for e in batch)
            # total_authors = sum(len(e.get("authors", [])) for e in batch)
            total_authors = sum(
                1 for e in batch for a in e.get("authors", [])
                if a.get("LastName") and a.get("ForeName") and a.get("organization") and a.get("country")
            )

            aff_seq = get_next_seq_iter(conn, "affiliation_no_seq", total_affiliations)
            author_seq = get_next_seq_iter(conn, "author_no_seq", total_authors)

            affiliation_inserts = []
            author_inserts = []
            author_aff_inserts = []
            
            seen_affiliations = dict()  # (pmid, affiliation_text) → aff_no
            
            for entry in batch:
                pmid = entry["pmid"]
                authors = entry.get("authors", [])

                # insert 대상만 필터링
                filtered_authors = []
                for a in authors:
                    lastname = a.get("LastName")
                    forename = a.get("ForeName")
                    org = a.get("organization")
                    country = a.get("country")

                    if not (lastname and forename and org and country):
                        continue

                    key = (pmid, lastname.strip().upper(), forename.strip().upper())
                    if key in seen_authors:
                        continue
                    seen_authors.add(key)

                    filtered_authors.append((a, key))

                # insert할 사람 수만큼 author_no 뽑음
                author_seq = get_next_seq_iter(conn, "author_no_seq", len(filtered_authors))

                for (author, key), author_no in zip(filtered_authors, author_seq):
                    aff_text = author.get("affiliation_text")
                    org = author.get("organization")
                    country = author.get("country")
                    lastname = author.get("LastName")
                    forename = author.get("ForeName")

                    # affiliation insert도 여기서 처리
                    aff_key = (pmid, aff_text.strip())
                    if aff_key in seen_affiliations:
                        aff_no = seen_affiliations[aff_key]
                    else:
                        aff_no = next(aff_seq)
                        seen_affiliations[aff_key] = aff_no
                        affiliation_inserts.append({
                            "pmid": pmid,
                            "affiliation_no": aff_no,
                            "affiliation": aff_text,
                            "affiliation_country": country,
                            "affiliation_organization": org,
                            "affiliation_identifier_source": None,
                            "affiliation_identifier": None
                        })

                    author_inserts.append({
                        "pmid": pmid,
                        "author_no": author_no,
                        "valid_yn": "Y",
                        "lastname": lastname,
                        "forename": forename,
                        "collectivename": author.get("CollectiveName"),
                        "initials": author.get("Initials"),
                        "suffix": author.get("Suffix"),
                        "identifier_source": author.get("IdentifierSource"),
                        "identifier": author.get("Identifier")
                    })

                    author_aff_inserts.append({
                        "pmid": pmid,
                        "author_no": author_no,
                        "affiliation_no": aff_no
                    })

            # affiliation insert
            memory_guard(max_gb=130)
            if affiliation_inserts:
                chunked_insert(conn, metadata.tables["pub_affiliation"], affiliation_inserts)
                count["affiliation"] += len(affiliation_inserts)

            # author insert
            memory_guard(max_gb=130)
            if author_inserts:
                chunked_insert(conn, metadata.tables["pub_author"], author_inserts)
                count["author"] += len(author_inserts)

            # author-affiliation insert
            memory_guard(max_gb=130)
            if author_aff_inserts:
                chunked_insert(conn, metadata.tables["pub_author_affiliation"], author_aff_inserts)
                count["author_aff"] += len(author_aff_inserts)
            # 5) keywords
            total_keywords = sum(len(e["keywords"]) for e in batch)
            keyword_seq = get_next_seq_batch(conn, "keyword_index", total_keywords)
            kw_idx = 0
            keyword_inserts = []
            for entry in batch:
                pmid = entry["pmid"]
                for kw in entry["keywords"]:
                    keyword_inserts.append({
                        "keyword_index": keyword_seq[kw_idx],
                        "pmid": pmid,
                        "keyword": kw,
                        "keyword_major_topic_yn": "N"
                    })
                    kw_idx += 1
            if keyword_inserts:
                conn.execute(metadata.tables["pub_keyword_list"].insert().prefix_with("IGNORE"), keyword_inserts)
                count["keyword"] += len(keyword_inserts)
            print("references Insert 전 메모리체크")
            memory_guard(max_gb=130)
            # 6) references
            total_references = sum(len(e["references"]) for e in batch)
            ref_seq = get_next_seq_batch(conn, "ref_no_seq", total_references)
            ref_idx = 0
            reference_inserts = []
            for entry in batch:
                pmid = entry["pmid"]
                for ref in entry["references"]:
                    reference_inserts.append({
                        "pmid": pmid,
                        "ref_no": ref_seq[ref_idx],
                        "citation": ref.get("citation"),
                        "ref_article_id_pubmed": ref.get("pubmed"),
                        "ref_article_id_doi": ref.get("doi"),
                        "ref_article_id_pmcid": ref.get("pmcid"),
                        "ref_article_id_pii": ref.get("pii")
                    })
                    ref_idx += 1
            if reference_inserts:
                conn.execute(metadata.tables["pub_reference"].insert().prefix_with("IGNORE"), reference_inserts)
                count["reference"] += len(reference_inserts)
            print("mesh Insert 전 메모리체크")
            memory_guard(max_gb=130)
            # 7) mesh
            total_mesh = sum(len(e["mesh"]) for e in batch)
            desc_seq = get_next_seq_batch(conn, "descriptor_index", total_mesh)
            desc_idx = 0
            mesh_inserts = []
            for entry in batch:
                pmid = entry["pmid"]
                for mesh in entry["mesh"]:
                    mesh_inserts.append({
                        "pmid": pmid,
                        "descriptor_index": desc_seq[desc_idx],
                        "descriptor_name": mesh.get("DescriptorName"),
                        "descriptor_major_topic_yn": mesh.get("DescriptorMajorTopicYN", "N"),
                        "descriptor_type": "MESH",
                        "descriptor_ui": mesh.get("DescriptorUI"),
                        "qualifier_name": mesh.get("QualifierName"),
                        "qualifer_major_topic_yn": mesh.get("QualifierMajorTopicYN"),
                        "qualifier_ui": mesh.get("QualifierUI")
                    })
                    desc_idx += 1
            if mesh_inserts:
                conn.execute(metadata.tables["pub_mesh_heading"].insert().prefix_with("IGNORE"), mesh_inserts)
                count["mesh"] += len(mesh_inserts)
            print("journal Insert 전 메모리체크")
            memory_guard(max_gb=130)
            # 8) journal
            journal_inserts = []
            for entry in batch:
                journal_title = entry.get("journal_title")
                journal_abbr = entry.get("journal_iso_abbreviation")
                journal_volume = entry.get("journal_volume")
                journal_issue = entry.get("journal_issue")
                journal_year = entry.get("journal_pub_year")
                journal_month = entry.get("journal_pub_month")
                journal_day = entry.get("journal_pub_day")
                if any([journal_title, journal_abbr, journal_volume, journal_issue, journal_year, journal_month, journal_day]):
                    journal_inserts.append({
                        "pmid": entry["pmid"],
                        "title": journal_title,
                        "iso_abbreviation": journal_abbr,
                        "volume": journal_volume,
                        "issue": journal_issue,
                        "pub_year": journal_year,
                        "pub_month": journal_month,
                        "pub_day": journal_day
                    })
            if journal_inserts:
                conn.execute(metadata.tables["pub_journal"].insert().prefix_with("IGNORE"), journal_inserts)
                count["journal"] += len(journal_inserts)
            print("databank Insert 전 메모리체크")
            memory_guard(max_gb=130)
            # 9) databanks
            databank_inserts = []
            for entry in batch:
                pmid = entry["pmid"]
                for db_entry in entry.get("databanks", []):
                    databank_inserts.append({
                        "pmid": pmid,
                        "databank_list_complete_yn": db_entry.get("complete_yn"),
                        "databank_name": db_entry.get("databank_name"),
                        "accession_number": db_entry.get("accession_number")
                    })
            if databank_inserts:
                conn.execute(metadata.tables["pub_databank_list"].insert().prefix_with("IGNORE"), databank_inserts)
                count["databank"] += len(databank_inserts)

        print(f" Batch {batch_start}~{batch_end - 1} 완료")

    print("\n 삽입 결과 요약:")
    for k, v in count.items():
        print(f" {k}: {v}건")
    print(" DB Insert 완료!\n")

    memory_guard(max_gb=130)

# 순차 처리 루프
def sequential_parse_with_progress(xml_path, db_url, total_articles=1000, chunk_size=1000, filename=None):
    success_count = 0
    fail_count_missing_affiliation = 0
    fail_count_empty_org_country = 0

    start_time = time.time()

    for offset in tqdm(range(0, total_articles, chunk_size), desc=f"📦 Parsing [{filename or os.path.basename(xml_path)}]"):
        chunk = parse_pubmed_chunk(xml_path, start_offset=offset, chunk_limit=chunk_size)

        parsed = []
        for item in chunk:
            reason = item.get("skip_reason", None)
            if reason == "missing_affiliation":
                fail_count_missing_affiliation += 1
                continue
            elif reason == "empty_org_country":
                fail_count_empty_org_country += 1
                continue

            success_count += 1
            parsed.append(item)

        if parsed:
            insert_all_to_db(parsed, engine, db_url)
            del parsed
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = time.time() - start_time
    print(f"\n 전체 파싱 소요 시간: {elapsed:.2f}초")
    print(f" 주소 식별 성공: {success_count}건")
    print(f" 스킵된 논문 (affiliation 없음): {fail_count_missing_affiliation}건")
    print(f" 스킵된 논문 (org/country 누락): {fail_count_empty_org_country}건")
    print(" 전체 완료!")

# PubMed XML 단일 파일 처리
def count_pubmed_articles(xml_path):
    count = 0
    context = ET.iterparse(xml_path, events=('end',), tag='PubmedArticle', recover=True, encoding='utf-8')
    for _, elem in context:
        count += 1
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
    gc.collect()
    return count

def process_pubmed_all(xml_path, db_url):
    print(" 전체 PubMed 처리 시작")
    total_articles = count_pubmed_articles(xml_path)
    filename = os.path.basename(xml_path)
    sequential_parse_with_progress(xml_path, db_url, total_articles=total_articles, chunk_size=1000, filename=filename)
    print(" 전체 완료!")

# 디렉토리 내 모든 파일 처리
def process_pubmed_dir(xml_dir, db_url, start_index=0):
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    for i, xml_file in enumerate(sorted(xml_files)):
        if i < start_index:
            continue  # 건너뛰기
        full_path = os.path.join(xml_dir, xml_file)
        print(f"\n📄 [{i+1}/{len(xml_files)}] Processing: {xml_file}")
        process_pubmed_all(xml_path=full_path, db_url=db_url)

# DB 초기화 함수
def init_engine(db_url):
    global engine, metadata
    engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600)
    metadata = MetaData()
    metadata.reflect(bind=engine)

def memory_guard(max_gb=130):
    import os, psutil, sys
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    print(f" 현재 메모리 사용량: {rss:.2f} GB / 제한: {max_gb} GB / 메모리 체크")
    if rss > max_gb:
        print(f" 메모리 {rss:.2f}GB 초과! 강제 종료")
        sys.exit(1)

# 메인 루틴
if __name__ == "__main__":
    db_url = "mysql+pymysql://mz:mediazen1%21@127.0.0.1:3306/pubmed"
    init_engine(db_url)
    init_spacy()
    process_pubmed_dir(
        xml_dir="/home01/bdata-webportal/pubmed/xml",
        db_url=db_url,
        start_index=0,
        # start_index=0,
    )
