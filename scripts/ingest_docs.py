from api.rag.index import build_or_update_index

if __name__ == "__main__":
    print("Building/Updating Chroma index from ./docs ...")
    build_or_update_index("docs")
    print("Done.")
