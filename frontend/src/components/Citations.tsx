import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Box from "@mui/material/Box";
import Chip from "@mui/material/Chip";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import FormatQuoteIcon from "@mui/icons-material/FormatQuote";

import type { Citation } from "../api/types";

interface CitationsProps {
  citations: Citation[];
}

/** Expandable list of the verified citations backing a grounded answer. */
export function Citations({ citations }: CitationsProps) {
  if (citations.length === 0) {
    return null;
  }

  return (
    <Accordion
      disableGutters
      elevation={0}
      sx={{
        mt: 1,
        bgcolor: "transparent",
        "&:before": { display: "none" },
      }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{ minHeight: 0, px: 0, "& .MuiAccordionSummary-content": { my: 0.5 } }}
      >
        <Typography variant="caption" color="text.secondary">
          {citations.length} source{citations.length === 1 ? "" : "s"}
        </Typography>
      </AccordionSummary>
      <AccordionDetails sx={{ px: 0, pt: 0 }}>
        <Stack spacing={1}>
          {citations.map((c) => (
            <Box
              key={`${c.source}#${c.chunk_index}:${c.marker}`}
              sx={{
                borderLeft: "3px solid",
                borderColor: "primary.light",
                pl: 1.5,
                py: 0.5,
              }}
            >
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
              <Typography
                variant="body2"
                sx={{ mt: 0.5, fontStyle: "italic", color: "text.secondary" }}
              >
                <FormatQuoteIcon sx={{ fontSize: 14, mr: 0.5, opacity: 0.6 }} />
                {c.quote}
              </Typography>
            </Box>
          ))}
        </Stack>
      </AccordionDetails>
    </Accordion>
  );
}
