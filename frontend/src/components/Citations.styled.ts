import { styled } from "@mui/material/styles";
import Accordion from "@mui/material/Accordion";
import AccordionSummary from "@mui/material/AccordionSummary";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

/** Flush, borderless accordion wrapping the citation list. */
export const CitationAccordion = styled(Accordion)(({ theme }) => ({
  marginTop: theme.spacing(1),
  backgroundColor: "transparent",
  "&:before": { display: "none" },
}));

/** Compact summary row ("N sources") with no gutters. */
export const CitationSummary = styled(AccordionSummary)(({ theme }) => ({
  minHeight: 0,
  paddingInline: 0,
  "& .MuiAccordionSummary-content": { marginBlock: theme.spacing(0.5) },
}));

/** A single citation entry with an accent left border. */
export const CitationItem = styled(Box)(({ theme }) => ({
  borderLeft: "3px solid",
  borderColor: theme.palette.primary.light,
  paddingLeft: theme.spacing(1.5),
  paddingBlock: theme.spacing(0.5),
}));

/** The italic, muted quote text of a citation. */
export const Quote = styled(Typography)(({ theme }) => ({
  marginTop: theme.spacing(0.5),
  fontStyle: "italic",
  color: theme.palette.text.secondary,
}));
