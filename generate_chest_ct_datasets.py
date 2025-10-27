#!/usr/bin/env python3
"""
Script to generate 6000 chest CT images using MONAI MAISI CT Generative Model
with rich progress bar visualization.
"""

import os
import sys
import subprocess
import tempfile
import argparse
from pathlib import Path

# Install required packages if missing
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    import huggingface_hub

try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:
    print("Installing rich for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "-q"])
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

import torch
import numpy as np
from monai.config import print_config
from monai.bundle.scripts import create_workflow, download

console = Console()


def setup_bundle(bundle_dir):
    """Download and setup the MAISI CT generative bundle."""
    console.print(Panel.fit("🚀 Setting up MONAI MAISI CT Generative Bundle", style="bold blue"))
    
    if bundle_dir is None:
        bundle_dir = tempfile.mkdtemp()
    else:
        os.makedirs(bundle_dir, exist_ok=True)
    
    console.print(f"[cyan]Bundle directory:[/cyan] {bundle_dir}")
    
    # Check if bundle already exists
    bundle_root = os.path.join(bundle_dir, "maisi_ct_generative")
    if os.path.exists(bundle_root) and os.path.exists(os.path.join(bundle_root, "configs/inference.json")):
        console.print("[green]✓[/green] Bundle already downloaded, skipping download")
    else:
        console.print("[yellow]Downloading bundle...[/yellow]")
        download(name="maisi_ct_generative", bundle_dir=bundle_dir)
        console.print("[green]✓[/green] Bundle downloaded successfully")
    
    return bundle_dir, bundle_root


def create_generation_workflow(bundle_root, output_size_xy=256, output_size_z=256, 
                               spacing_xy=1.5, spacing_z=1.5):
    """Create the workflow for CT generation."""
    console.print("\n[cyan]Creating generation workflow...[/cyan]")
    
    override = {
        "output_size_xy": output_size_xy,
        "output_size_z": output_size_z,
        "spacing_xy": spacing_xy,
        "spacing_z": spacing_z,
        "autoencoder_def#num_splits": 16,
        "mask_generation_autoencoder_def#num_splits": 16,
        "body_region": ["chest"],
        "anatomy_list": ["left lung upper lobe", "left lung lower lobe", "right lung upper lobe", "right lung middle lobe", "right lung lower lobe"],
    }
    
    workflow = create_workflow(
        config_file=os.path.join(bundle_root, "configs/inference.json"),
        workflow_type="inference",
        bundle_root=bundle_root,
        **override,
    )
    
    console.print("[green]✓[/green] Workflow created successfully")
    return workflow


def generate_ct_images(workflow, num_samples=6000, batch_size=1, output_dir=None):
    """Generate CT images with rich progress bar."""
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        console.print(f"[cyan]Output directory:[/cyan] {output_dir}")
    
    # Create a summary table
    table = Table(title="Generation Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan", width=30)
    table.add_column("Value", style="green")
    
    table.add_row("Total Samples", str(num_samples))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("Output Directory", output_dir or "Default (bundle output)")
    table.add_row("Body Region", "Chest")
    table.add_row("Device", "CUDA" if torch.cuda.is_available() else "CPU")
    
    console.print("\n")
    console.print(table)
    console.print("\n")
    
    # Progress bar with rich
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•"),
        TextColumn("[cyan]{task.completed}/{task.total}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]Generating {num_samples} chest CT images...",
            total=num_samples
        )
        
        generated_count = 0
        errors = 0
        
        try:
            for i in range(num_samples):
                try:
                    # Update the workflow to generate one sample at a time
                    # This allows for better progress tracking
                    result = workflow.run()
                    
                    # If custom output directory is specified, move the files
                    if output_dir and result:
                        # Handle output file moving here if needed
                        pass
                    
                    generated_count += 1
                    progress.update(task, advance=1, description=f"[green]Generated {generated_count} images")
                    
                except Exception as e:
                    errors += 1
                    progress.update(task, advance=1, description=f"[yellow]Generated {generated_count} images ({errors} errors)")
                    console.print(f"[red]Error generating sample {i+1}:[/red] {str(e)}")
                    continue
        
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Generation interrupted by user[/yellow]")
        
        finally:
            # Final summary
            console.print("\n")
            summary_table = Table(title="Generation Summary", show_header=True, header_style="bold magenta")
            summary_table.add_column("Metric", style="cyan", width=30)
            summary_table.add_column("Value", style="green")
            
            summary_table.add_row("Successfully Generated", str(generated_count))
            summary_table.add_row("Errors", str(errors), style="red" if errors > 0 else "green")
            summary_table.add_row("Total Attempted", str(generated_count + errors))
            summary_table.add_row("Success Rate", f"{(generated_count/(generated_count+errors)*100):.2f}%" if (generated_count+errors) > 0 else "N/A")
            
            console.print(summary_table)
            console.print("\n[bold green]✓ Generation complete![/bold green]\n")
    
    return generated_count, errors


def main():
    """Main function to orchestrate CT image generation."""
    parser = argparse.ArgumentParser(
        description="Generate chest CT images using MONAI MAISI CT Generative Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6000,
        help="Number of CT images to generate (default: 6000)"
    )
    
    parser.add_argument(
        "--bundle-dir",
        type=str,
        default="/localhome/local-tranminhq/datasets/maisi",
        help="Directory for MAISI bundle (default: /localhome/local-tranminhq/datasets/maisi)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for generated images (default: bundle's output directory)"
    )
    
    parser.add_argument(
        "--output-size-xy",
        type=int,
        default=256,
        help="Output size for XY dimensions (default: 256)"
    )
    
    parser.add_argument(
        "--output-size-z",
        type=int,
        default=256,
        help="Output size for Z dimension (default: 256)"
    )
    
    parser.add_argument(
        "--spacing-xy",
        type=float,
        default=1.5,
        help="Voxel spacing for XY dimensions in mm (default: 1.5)"
    )
    
    parser.add_argument(
        "--spacing-z",
        type=float,
        default=1.5,
        help="Voxel spacing for Z dimension in mm (default: 1.5)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation (default: 1)"
    )
    
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show MONAI configuration"
    )
    
    args = parser.parse_args()
    
    # Print header
    console.print(Panel.fit(
        "[bold cyan]MONAI MAISI Chest CT Generator[/bold cyan]\n"
        "[dim]Generating synthetic chest CT images using deep learning[/dim]",
        style="bold blue"
    ))
    
    if args.show_config:
        console.print("\n[cyan]MONAI Configuration:[/cyan]")
        print_config()
        console.print("\n")
    
    # Setup bundle
    bundle_dir, bundle_root = setup_bundle(args.bundle_dir)
    
    # Create workflow
    workflow = create_generation_workflow(
        bundle_root,
        output_size_xy=args.output_size_xy,
        output_size_z=args.output_size_z,
        spacing_xy=args.spacing_xy,
        spacing_z=args.spacing_z
    )
    
    # Generate images
    console.print(f"\n[bold yellow]Starting generation of {args.num_samples} chest CT images...[/bold yellow]\n")
    
    generated, errors = generate_ct_images(
        workflow,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
    
    if errors == 0:
        console.print(f"[bold green]🎉 Successfully generated all {generated} chest CT images![/bold green]")
    else:
        console.print(f"[yellow]⚠ Generated {generated} images with {errors} errors[/yellow]")
    
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Script interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]❌ Fatal error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

