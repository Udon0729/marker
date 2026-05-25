import os

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
    "1"  # Transformers uses .isin for a simple op, which is not supported on MPS
)

import time
import click

from marker.config.parser import ConfigParser
from marker.config.printer import CustomClickPrinter
from marker.logger import configure_logging, get_logger
from marker.models import create_model_dict
from marker.output import save_output
from marker.utils.url import download_pdf_to_memory, filename_from_url, is_url

configure_logging()
logger = get_logger()


@click.command(
    cls=CustomClickPrinter,
    help="Convert a single PDF to markdown. `fpath` accepts a local file path or an http(s) URL pointing to a PDF.",
)
@click.argument("fpath", type=str)
@ConfigParser.common_options
def convert_single_cli(fpath: str, **kwargs):
    models = create_model_dict()
    start = time.time()

    converter_input = fpath
    if is_url(fpath):
        logger.info(f"Downloading PDF from {fpath} into memory")
        converter_input, derived_name = download_pdf_to_memory(fpath)
        # Make sure the provider's display name matches the URL-derived filename
        # so debug paths, logs, and Document.filepath stay informative even
        # though the PDF never touches disk.
        kwargs["source_label_override"] = derived_name

    config_parser = ConfigParser(kwargs)

    converter_cls = config_parser.get_converter_cls()
    converter = converter_cls(
        config=config_parser.generate_config_dict(),
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(converter_input)
    out_folder = config_parser.get_output_folder(fpath)
    save_output(rendered, out_folder, config_parser.get_base_filename(fpath))

    logger.info(f"Saved markdown to {out_folder}")
    logger.info(f"Total time: {time.time() - start}")
