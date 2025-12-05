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

# 파일 경로
ORG_EXCEL_PATH = "/home/mediazen/pubmed_temp_py/files/20250704_084016/organization_리스트/organization_ror167_250703.xlsx"
SUB_ORG_EXCEL_PATH = "/home/mediazen/pubmed_temp_py/files/20250704_084016/organization_리스트/sub_organization_250703.xlsx"

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
        print("✅ GPU 사용: ", spacy.prefer_gpu())
    else:
        spacy.require_cpu()
        print("⚠️ GPU 사용 불가 → CPU 처리")
    GLOBAL_NLP_DATA = setup_spacy_and_dict_with_ror(ORG_EXCEL_PATH, SUB_ORG_EXCEL_PATH)

# 배치 주소 정제 함수
def batch_refine_affiliations(address_list, nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map):
    docs = nlp.pipe(address_list, batch_size=64)
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

# PubMed XML 파싱 + 배치 NER 적용
def parse_pubmed_chunk(xml_path, start_offset=0, chunk_limit=1000):
    global GLOBAL_NLP_DATA
    nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map = GLOBAL_NLP_DATA
    results = []
    context = ET.iterparse(xml_path, events=('end',), tag='PubmedArticle', recover=True, encoding='utf-8')
    temp_chunk = []

    for i, (_, elem) in enumerate(context):
        if i < start_offset:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            continue

        pmid = elem.findtext(".//PMID")

        # 날짜 처리
        pubdate = elem.find(".//PubDate")
        year = pubdate.findtext("Year") if pubdate is not None else None
        month = pubdate.findtext("Month") if pubdate is not None else None
        day = pubdate.findtext("Day") if pubdate is not None else None

        # 제목
        article_title = elem.findtext(".//ArticleTitle")

        # 키워드
        keywords = [k.text for k in elem.findall(".//KeywordList/Keyword") if k.text]

        # Journal 정보 추출
        journal_elem = elem.find(".//Journal")
        journal_title = journal_elem.findtext("Title") if journal_elem is not None else None
        journal_iso_abbreviation = journal_elem.findtext("ISOAbbreviation") if journal_elem is not None else None
        journal_volume = elem.findtext(".//JournalIssue/Volume")
        journal_issue = elem.findtext(".//JournalIssue/Issue")
        journal_pubdate = elem.find(".//JournalIssue/PubDate")
        journal_pub_year = journal_pubdate.findtext("Year") if journal_pubdate is not None else None
        journal_pub_month = journal_pubdate.findtext("Month") if journal_pubdate is not None else None
        journal_pub_day = journal_pubdate.findtext("Day") if journal_pubdate is not None else None

        # DataBank 정보 추출
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

        # 저자
        authors = []
        for a in elem.findall(".//AuthorList/Author"):
            authors.append({
                "LastName": a.findtext("LastName"),
                "ForeName": a.findtext("ForeName"),
                "Initials": a.findtext("Initials"),
                "Suffix": a.findtext("Suffix"),
                "CollectiveName": a.findtext("CollectiveName"),
                "Identifier": a.findtext("Identifier"),
                "IdentifierSource": a.find("Identifier").attrib.get("Source") if a.find("Identifier") is not None else None
            })

        # 소속
        affiliations = [a.text for a in elem.findall(".//Affiliation") if a.text]

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

        temp_chunk.append((pmid, affiliations, keywords, references, mesh_terms, year, month, day, article_title, authors))
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

        if len(temp_chunk) >= chunk_limit:
            break

    # 주소 정제
    all_affs = [aff for entry in temp_chunk for aff in entry[1]]
    refined_affs = batch_refine_affiliations(all_affs, nlp, clean_org, clean_sub_org, raw_org, raw_sub_org, org_country_map)

    idx = 0
    for entry in temp_chunk:
        aff_list = refined_affs[idx:idx + len(entry[1])]
        idx += len(entry[1])
        results.append({
            "pmid": entry[0],
            "affiliations": aff_list,
            "keywords": entry[2],
            "references": entry[3],
            "mesh": entry[4],
            "year": entry[5],
            "month": entry[6],
            "day": entry[7],
            "article_title": entry[8],
            "authors": entry[9],
            "journal_title": journal_title,
            "journal_iso_abbreviation": journal_iso_abbreviation,
            "journal_volume": journal_volume,
            "journal_issue": journal_issue,
            "journal_pub_year": journal_pub_year,
            "journal_pub_month": journal_pub_month,
            "journal_pub_day": journal_pub_day,
            "databanks": databanks,
        })

    return results


# DB Insert 함수

def get_next_seq(conn, seq_name):
    return conn.execute(text(f"SELECT NEXT VALUE FOR {seq_name}")).scalar()

def insert_all_to_db(parsed_data, db_url):
    engine = create_engine(db_url)
    metadata = MetaData()
    metadata.reflect(bind=engine)

    count_pmid = count_title = count_pubdate = count_affiliation = count_author = count_author_aff = count_keyword = count_reference = count_mesh = count_journal = count_databank = 0

    with engine.begin() as conn:
        for entry in tqdm(parsed_data, desc="🛠️ Inserting to DB"):
            pmid = entry["pmid"]
            year, month, day = entry["year"], entry["month"], entry["day"]
            article_title = entry["article_title"]

            conn.execute(mysql_insert(metadata.tables["pub_pmid_list"]).values(pmid=pmid).prefix_with("IGNORE"))
            count_pmid += 1

            if article_title:
                conn.execute(mysql_insert(metadata.tables["pub_article_title"]).values(
                    pmid=pmid, article_title=article_title).prefix_with("IGNORE"))
                count_title += 1

            if year:
                conn.execute(mysql_insert(metadata.tables["pub_pubdate"]).values(
                    pmid=pmid, year=year, month=month, day=day).prefix_with("IGNORE"))
                count_pubdate += 1

            for aff, org, country in entry["affiliations"]:
                aff_no = get_next_seq(conn, "affiliation_no_seq")
                conn.execute(mysql_insert(metadata.tables["pub_affiliation"]).values(
                    pmid=pmid, affiliation_no=aff_no, affiliation=aff,
                    affiliation_country=country, affiliation_organization=org,
                    affiliation_identifier_source=None, affiliation_identifier=None
                ).prefix_with("IGNORE"))
                count_affiliation += 1

                for author in entry.get("authors", []):
                    author_no = get_next_seq(conn, "author_no_seq")
                    conn.execute(mysql_insert(metadata.tables["pub_author"]).values(
                        pmid=pmid, author_no=author_no, valid_yn="Y",
                        lastname=author.get("LastName"),
                        forename=author.get("ForeName"),
                        collectivename=author.get("CollectiveName"),
                        initials=author.get("Initials"),
                        suffix=author.get("Suffix"),
                        identifier_source=author.get("IdentifierSource"),
                        identifier=author.get("Identifier")
                    ).prefix_with("IGNORE"))
                    count_author += 1

                    conn.execute(mysql_insert(metadata.tables["pub_author_affiliation"]).values(
                        pmid=pmid, author_no=author_no, affiliation_no=aff_no
                    ).prefix_with("IGNORE"))
                    count_author_aff += 1

            for kw in entry["keywords"]:
                kw_idx = get_next_seq(conn, "keyword_index")
                conn.execute(mysql_insert(metadata.tables["pub_keyword_list"]).values(
                    keyword_index=kw_idx, pmid=pmid, keyword=kw, keyword_major_topic_yn="N"
                ).prefix_with("IGNORE"))
                count_keyword += 1

            for ref in entry["references"]:
                ref_no = get_next_seq(conn, "ref_no_seq")
                conn.execute(mysql_insert(metadata.tables["pub_reference"]).values(
                    pmid=pmid, ref_no=ref_no,
                    citation=ref.get("citation"),
                    ref_article_id_pubmed=ref.get("pubmed"),
                    ref_article_id_doi=ref.get("doi"),
                    ref_article_id_pmcid=ref.get("pmcid"),
                    ref_article_id_pii=ref.get("pii")
                ).prefix_with("IGNORE"))
                count_reference += 1

            for mesh in entry["mesh"]:
                desc_idx = get_next_seq(conn, "descriptor_index")
                conn.execute(mysql_insert(metadata.tables["pub_mesh_heading"]).values(
                    pmid=pmid, descriptor_index=desc_idx,
                    descriptor_name=mesh.get("DescriptorName"),
                    descriptor_major_topic_yn=mesh.get("DescriptorMajorTopicYN", "N"),
                    descriptor_type="MESH",
                    descriptor_ui=mesh.get("DescriptorUI"),
                    qualifier_name=mesh.get("QualifierName"),
                    qualifer_major_topic_yn=mesh.get("QualifierMajorTopicYN"),
                    qualifier_ui=mesh.get("QualifierUI")
                ).prefix_with("IGNORE"))
                count_mesh += 1

            journal_title = entry.get("journal_title")
            journal_abbr = entry.get("journal_iso_abbreviation")
            journal_volume = entry.get("journal_volume")
            journal_issue = entry.get("journal_issue")
            journal_year = entry.get("journal_pub_year")
            journal_month = entry.get("journal_pub_month")
            journal_day = entry.get("journal_pub_day")

            if any([journal_title, journal_abbr, journal_volume, journal_issue, journal_year, journal_month, journal_day]):
                conn.execute(mysql_insert(metadata.tables["pub_journal"]).values(
                    pmid=pmid,
                    title=journal_title,
                    iso_abbreviation=journal_abbr,
                    volume=journal_volume,
                    issue=journal_issue,
                    pub_year=journal_year,
                    pub_month=journal_month,
                    pub_day=journal_day
                ).prefix_with("IGNORE"))
            count_journal += 1

            for db_entry in entry.get("databanks", []):
                conn.execute(mysql_insert(metadata.tables["pub_databank_list"]).values(
                    pmid=pmid,
                    databank_list_complete_yn=db_entry.get("complete_yn"),
                    databank_name=db_entry.get("databank_name"),
                    accession_number=db_entry.get("accession_number")
                ).prefix_with("IGNORE"))
            count_databank += 1

    print("\n📊 삽입 결과 요약:")
    print(f"🧾 pub_pmid_list: {count_pmid}건")
    print(f"🧾 pub_article_title: {count_title}건")
    print(f"🧾 pub_pubdate: {count_pubdate}건")
    print(f"🧾 pub_affiliation: {count_affiliation}건")
    print(f"🧾 pub_author: {count_author}건")
    print(f"🧾 pub_author_affiliation: {count_author_aff}건")
    print(f"🧾 pub_keyword_list: {count_keyword}건")
    print(f"🧾 pub_reference: {count_reference}건")
    print(f"🧾 pub_mesh_heading: {count_mesh}건")
    print(f"🧾 pub_journal: {count_journal}건")
    print(f"🧾 pub_databank_list: {count_databank}건")
    print("✅ DB Insert 완료!\n")

# 순차 처리 루프
def sequential_parse_with_progress(xml_path, total_articles=1000, chunk_size=100):
    success_count, fail_count = 0, 0
    results = []
    start_time = time.time()
    for offset in tqdm(range(0, total_articles, chunk_size), desc="📦 Parsing XML chunks (GPU 배치)"):
        chunk = parse_pubmed_chunk(xml_path, start_offset=offset, chunk_limit=chunk_size)
        for item in chunk:
            if any(org for _, org, _ in item["affiliations"]):
                success_count += 1
                results.append(item)
            else:
                fail_count += 1
    elapsed = time.time() - start_time
    print(f"⏱️ 전체 파싱 소요 시간: {elapsed:.2f}초")
    print(f"✅ 주소 식별 성공: {success_count}건 / ❌ 실패: {fail_count}건")
    return results

# PubMed XML 단일 파일 처리
def count_pubmed_articles(xml_path):
    count = 0
    context = ET.iterparse(xml_path, events=('end',), tag='PubmedArticle', recover=True, encoding='utf-8')
    for _, elem in context:
        count += 1
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return count

def process_pubmed_all(xml_path, db_url):
    print("🚀 전체 PubMed 처리 시작")
    total_articles = count_pubmed_articles(xml_path)
    parsed = sequential_parse_with_progress(xml_path, total_articles=total_articles, chunk_size=100)
    insert_all_to_db(parsed, db_url)
    print("✅ 전체 완료!")

# 디렉토리 내 모든 파일 처리
def process_pubmed_dir(xml_dir, db_url):
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    for i, xml_file in enumerate(sorted(xml_files)):
        full_path = os.path.join(xml_dir, xml_file)
        print(f"\n📄 [{i+1}/{len(xml_files)}] Processing: {xml_file}")
        process_pubmed_all(xml_path=full_path, db_url=db_url)

# 메인 루틴
if __name__ == "__main__":
    init_spacy()
    process_pubmed_dir(
        xml_dir="/home01/bdata-webportal/pubmed/xml",
        db_url="mysql+pymysql://mz:mediazen1%21@127.0.0.1:3306/pubmed",
    )
