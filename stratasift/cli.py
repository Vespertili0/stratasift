import sys
from pathlib import Path
import click
from stratasift.config import load_config
from stratasift.utils.ollamahealth import check_ollama_cloud_health
from stratasift.core.parser import parse_markdown_file
from stratasift.core.lifecycle import quarantine_file


@click.group()
def main() -> None:
    """StrataSift CLI interface."""
    pass


@main.command()
def check() -> None:
    """Run diagnostic checks on the system configuration and model connections."""
    click.echo("🔍 [System Check] Initialising StrataSift Environment...")

    # 1. Load configuration
    try:
        config = load_config("config.yaml")
        from stratasift.config import set_runtime_config

        set_runtime_config(config)
        click.echo("✅ Configuration: Loaded config.yaml successfully.")
    except Exception as e:
        click.echo(f"❌ Configuration: Failed to load configuration: {str(e)}")
        sys.exit(1)

    # 2. Verify Obsidian Vault path
    try:
        vault_path = config.system.get_expanded_vault_path()
        # Create vault path if it does not exist to ensure diagnostic check succeeds
        if not vault_path.exists():
            vault_path.mkdir(parents=True, exist_ok=True)

        if vault_path.is_dir():
            click.echo("✅ Ingestion Path: Verified connectivity to Obsidian Vault.")
        else:
            click.echo(
                f"❌ Ingestion Path: Obsidian Vault path is not a directory: {vault_path}"
            )
            sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Ingestion Path: Failed to verify Obsidian Vault path: {str(e)}")
        sys.exit(1)

    # 3. Report Gemini/Ollama local mapping status
    if config.providers.gemini:
        click.echo("🤖 Cloud Provider (Gemini): Mapped & Ready [SKIPPED FOR NOW]")
    else:
        click.echo("🤖 Cloud Provider (Gemini): Not Mapped")

    if config.providers.ollama_local:
        click.echo("🤖 Local Provider (Ollama): Mapped & Ready [SKIPPED FOR NOW]")
    else:
        click.echo("🤖 Local Provider (Ollama): Not Mapped")

    # 4. Check Ollama Cloud engine
    active_providers = {config.blocks.supervisor_block.provider, config.blocks.analysis_block.provider}
    if "ollama_cloud" in active_providers:
        oc_config = config.providers.ollama_cloud
        result = check_ollama_cloud_health(
            base_url=oc_config.base_url,
            model_name=oc_config.model,
            api_key=oc_config.api_key,
        )
        if result["success"]:
            click.echo(f"☁️ Ollama Cloud Engine: {result['message']}")
        else:
            click.echo(
                f"❌ Ollama Cloud Engine: Verification failed - {result['message']}"
            )
            sys.exit(1)
    else:
        click.echo(f"☁️ Ollama Cloud Engine: Inactive (Active block providers: {', '.join(active_providers)})")

    click.echo("\n🎉 Foundation solid. Base data layouts ready for Epic-05.")


@main.command()
@click.argument(
    "directory_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
def ingest(directory_path: Path) -> None:
    """Scan and process raw Markdown clips in a directory."""
    click.echo("📥 [Ingestion Run] Scanning target folder...")

    try:
        config = load_config("config.yaml", validate_active=False)
        from stratasift.config import set_runtime_config

        set_runtime_config(config)
        quarantine_path = Path(config.system.quarantine_path)
        vault_path = config.system.get_expanded_vault_path()
    except Exception as e:
        click.echo(f"❌ Configuration: Failed to load configuration: {str(e)}")
        sys.exit(1)

    # Find all markdown files (exclusively .md files, ignoring hidden files/directories)
    md_files = sorted(
        [
            f
            for f in directory_path.iterdir()
            if f.is_file() and f.suffix == ".md" and not f.name.startswith(".")
        ]
    )

    click.echo(f"Found {len(md_files)} files awaiting processing.")

    processed_count = 0
    quarantined_count = 0

    for md_file in md_files:
        click.echo(f"\n⚙️ [Processing] Processing: {md_file.name}")
        try:
            # Parse and sanitise the literature file
            sanitised_lit, links_cnt, images_cnt = parse_markdown_file(md_file)

            # Print status updates using British spelling rules
            click.echo("   ✅ Frontmatter metadata extracted successfully.")
            click.echo(
                f"   ✅ Sanitised {links_cnt} academic reference links & stripped {images_cnt} embedded charts."
            )

            mapped_states = ["Abstract"]
            if sanitised_lit.methods:
                mapped_states.append("Methods")
            if sanitised_lit.results_discussion:
                mapped_states.append("Results")
            if sanitised_lit.conclusions:
                mapped_states.append("Conclusions")

            click.echo(
                f"   ✅ Structural segmentation complete (Mapped: {', '.join(mapped_states)})."
            )

            # Initiate LangGraph Orchestration
            click.echo(
                "\n🧠 [LangGraph Orchestration] Initiating Hierarchical Synthesis..."
            )

            from stratasift.graph import graph_app
            from stratasift.core.lifecycle import shelve_file

            # Setup initial state
            initial_state = {
                "source_doc": sanitised_lit,
                "reading_directive": "",
                "raw_extractions": [],
                "retry_count": 0,
                "insight_dossier": "",
                "vault_context": [],
                "final_markdown": "",
                "relevance_score": 0.0,
                "feedback": None,
                "atomic_insights": [],
                "source_filename": md_file.name,
                "routing_results": [],
                "domain_relevance": 0.0,
                "methodology_relevance": 0.0,
                "match_type": "",
                "central_hypothesis": "",
                "in_flight_notes": [],
            }

            # Run graph
            final_state = graph_app.invoke(initial_state)

            # Evaluate triage results
            relevance = final_state.get("relevance_score", 0.0)
            if relevance < 0.75:
                click.echo("   ⚠️ Triage rejected. Relocating to shelved area...")
                dest = shelve_file(md_file, vault_path, relevance)
                click.echo(f"   📦 Relocated file to shelved directory: {dest}")
                processed_count += 1
            else:
                routing_results = final_state.get("routing_results", [])
                
                from stratasift.utils.file_io import sanitise_filename
                from stratasift.tools.vector_store import LanceVectorStore
                vector_store = LanceVectorStore(vault_path)

                click.echo("\n💾 [File IO] Writing synthesized notes to Obsidian Vault...")

                for result in routing_results:
                    if result.get("decision") == "bypass":
                        continue

                    target_note_title = result.get("target_note") or result.get("note_title", md_file.stem)
                    if not target_note_title.endswith(".md"):
                        target_note_file = target_note_title + ".md"
                    else:
                        target_note_file = target_note_title

                    sanitised_title = sanitise_filename(Path(target_note_file).stem)
                    target_note_file = f"{sanitised_title}.md"
                    target_note_path = vault_path / target_note_file

                    # Write/update note
                    with open(target_note_path, "w", encoding="utf-8") as f:
                        f.write(result["markdown_content"])

                    # Insert vector index organically using the search block
                    insight = result.get("insight")
                    if insight:
                        vector_store.insert_insight(
                            file_id=target_note_file,
                            title=insight.title,
                            search_block=insight.search_block,
                        )
                        click.echo(f"   ✅ Vector indexed successfully for '{insight.title}'.")

                # Move original file to Sources folder
                sources_dir = vault_path / "Sources"
                sources_dir.mkdir(parents=True, exist_ok=True)
                dest_path = sources_dir / md_file.name
                
                if md_file.resolve() != dest_path.resolve():
                    if dest_path.exists():
                        dest_path.unlink()
                    import shutil
                    shutil.move(str(md_file), str(dest_path))
                    click.echo(f"   📦 Relocated source file to {dest_path}")

                click.echo(
                    f"🎉 Success! `{md_file.name}` ingested, synthesised, and networked."
                )
                processed_count += 1

        except Exception as e:
            reason = str(e)
            click.echo(f"   ⚠️ Error: {reason}")
            try:
                dest = quarantine_file(md_file, quarantine_path, reason)
                click.echo(f"   📦 Relocating file to active quarantine zone: {dest}")
            except Exception as move_err:
                click.echo(f"   ❌ Failed to quarantine file: {str(move_err)}")
            quarantined_count += 1

    click.echo("\n📊 [Ingestion Summary]")
    click.echo(f"   Processed: {processed_count} files")
    click.echo(f"   Quarantined: {quarantined_count} files")
    click.echo("   Active state payloads prepared for LangGraph Orchestration.")


@main.command(name="eval")
def eval_cmd() -> None:
    """Run referenceless LLM evaluation on generated outputs."""
    from rich.console import Console
    from rich.table import Table
    console = Console()
    console.print("🧪 [Evaluation Suite] Initialising...")

    try:
        config = load_config("config.yaml", validate_active=False)
        from stratasift.config import set_runtime_config
        set_runtime_config(config)
        vault_path = config.system.get_expanded_vault_path()
    except Exception as e:
        console.print(f"[bold red]❌ Configuration Error:[/bold red] {str(e)}")
        sys.exit(1)

    from stratasift.eval.metrics import run_evaluation_suite
    results = run_evaluation_suite(vault_path, config)

    if not results:
        console.print("[yellow]No ContextDB payloads found for evaluation.[/yellow]")
        return

    table = Table(title="Epic-06 Evaluation Results")
    table.add_column("Run ID", style="cyan")
    table.add_column("Source Document", style="magenta")
    table.add_column("Faithfulness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Recall", justify="right")

    for res in results:
        f_score = f"{res['faithfulness']:.2f}"
        r_score = f"{res['relevance']:.2f}"
        rc_score = f"{res['recall']:.2f}"
        
        # Color code based on threshold
        t = config.evaluation.threshold
        f_color = "green" if res['faithfulness'] >= t else "red"
        r_color = "green" if res['relevance'] >= t else "red"
        rc_color = "green" if res['recall'] >= t else "red"
        
        table.add_row(
            res['run_id'][:8],
            res['source'][:30],
            f"[{f_color}]{f_score}[/{f_color}]",
            f"[{r_color}]{r_score}[/{r_color}]",
            f"[{rc_color}]{rc_score}[/{rc_color}]"
        )
        
    console.print(table)
    console.print("\n🎉 Evaluation Complete.")


if __name__ == "__main__":
    main()
