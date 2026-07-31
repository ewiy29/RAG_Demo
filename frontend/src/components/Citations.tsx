import AccordionDetails from "@mui/material/AccordionDetails";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FormatQuoteIcon from "@mui/icons-material/FormatQuote";

import type { Citation } from "../api/types";
import {
  CitationAccordion,
  CitationItem,
  CitationSummary,
  Quote,
} from "./Citations.styled";

interface CitationsProps {
  citations: Citation[];
}

/** Expandable list of the verified citations backing a grounded answer. */
export function Citations({ citations }: CitationsProps) {
  if (citations.length === 0) {
    return null;
  }

  return (
    <CitationAccordion disableGutters elevation={0}>
      <CitationSummary expandIcon={<ExpandMoreIcon />}>
        <Typography variant="caption" color="text.secondary">
          {citations.length} source{citations.length === 1 ? "" : "s"}
        </Typography>
      </CitationSummary>
      <AccordionDetails sx={{ px: 0, pt: 0 }}>
        <Stack spacing={1}>
          {citations.map((c) => (
            <CitationItem key={`${c.source}#${c.chunk_index}:${c.marker}`}>
              <Stack
                direction="row"
                spacing={1}
                sx={{ alignItems: "center", flexWrap: "wrap" }}
              >
                <Chip
                  size="small"
                  label={`${c.source}#${c.chunk_index}`}
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  score {c.score.toFixed(3)}
                </Typography>
              </Stack>
              <Quote variant="body2">
                <FormatQuoteIcon sx={{ fontSize: 14, mr: 0.5, opacity: 0.6 }} />
                {c.quote}
              </Quote>
            </CitationItem>
          ))}
        </Stack>
      </AccordionDetails>
    </CitationAccordion>
  );
}
